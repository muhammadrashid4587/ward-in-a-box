"""Alert hooks: notify something outside the ward when a NEWS2
event fires.

A hook is configured per ward, off by default, in one of two ways:

* ``webhook_url`` — POST the alert payload as JSON to the URL. The
  POST runs in a daemon thread, so a slow or dead endpoint never
  blocks the ward tick; a failure is logged and counted, never
  fatal. Best effort: the ward records that the hook was invoked,
  not that the far end received it.
* ``callback`` — an in-process Python callable, called with the
  alert payload in the tick that raises the alert. A callback that
  raises is caught, logged, and counted as a failure — the tick
  still completes.

Two events use hooks. A trend alert (a NEWS2 score rising fast
enough to trip the jump or creep rule) fires the ward's
``trend_alert_hooks``; a deterioration episode opening (a
patient's NEWS2 risk leaving low) fires the ward's
``escalation_hooks``. Both share the same ``AlertHooks``
machinery — the same webhook and callback channels, the same
best-effort delivery — just a different event payload.

A fired event records whether the hook was invoked (``"hook_invoked"``
on the alert or episode, False when no hook is configured) and any
synchronous error (``"hook_errors"``). The early-warning summary
carries the ward totals — configured, invoked, failed — so a run
summary can say how many alerts reached the outside world. Hooks
never touch the vital-sign alarm list, and a ward with no hooks
configured behaves exactly as before.
"""

import json
import logging
import threading
import urllib.error
import urllib.request

log = logging.getLogger("ward.alert_hooks")

# Seconds a webhook POST may take before it is abandoned. The POST
# runs in a daemon thread, so this never stalls the ward — it only
# bounds how long the delivery thread lingers.
DEFAULT_WEBHOOK_TIMEOUT_S = 5.0


def _validate_hooks(value, kind, label):
    """Normalize a hook config for one event kind.

    None means no hooks (off); an ``AlertHooks`` passes through;
    a dict may set ``webhook_url`` (an http(s) URL to POST the
    alert payload to), ``callback`` (an in-process callable,
    Python API only — a scenario file cannot hold one), and
    ``timeout_s`` (a number above 0, how long a webhook POST may
    take). Returns an ``AlertHooks``.
    """
    if value is None:
        return AlertHooks(kind=kind)
    if isinstance(value, AlertHooks):
        return value
    if not isinstance(value, dict):
        raise ValueError("%s must be a dict, got %r" % (label, value))
    unknown = sorted(set(value) - {"webhook_url", "callback", "timeout_s"})
    if unknown:
        raise ValueError(
            "%s has unknown keys: %s" % (label, ", ".join(unknown))
        )
    return AlertHooks(
        webhook_url=value.get("webhook_url"),
        callback=value.get("callback"),
        timeout_s=value.get("timeout_s", DEFAULT_WEBHOOK_TIMEOUT_S),
        kind=kind,
    )


def validate_trend_alert_hooks(value):
    """Normalize a trend-alert hook config (see :func:`_validate_hooks`)."""
    return _validate_hooks(value, "trend-alert", "trend_alert_hooks")


def validate_escalation_hooks(value):
    """Normalize an escalation hook config (see :func:`_validate_hooks`).

    Fires when a deterioration episode opens — a patient's NEWS2
    risk leaving low — rather than when a trend alert fires.
    """
    return _validate_hooks(value, "escalation", "escalation_hooks")


class AlertHooks:
    """The configured hooks for one ward event, shared by its monitors.

    ``kind`` labels the event in log messages (``"trend-alert"``
    or ``"escalation"``); it changes nothing about delivery.
    """

    def __init__(self, webhook_url=None, callback=None,
                 timeout_s=DEFAULT_WEBHOOK_TIMEOUT_S, kind="trend-alert"):
        if webhook_url is not None:
            if not isinstance(webhook_url, str) or not webhook_url.startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    "alert hooks webhook_url must be an http(s) URL, "
                    "got %r" % (webhook_url,)
                )
        if callback is not None and not callable(callback):
            raise ValueError(
                "alert hooks callback must be callable, got %r"
                % (callback,)
            )
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError(
                "alert hooks timeout_s must be a number > 0, got %r"
                % (timeout_s,)
            )
        self.webhook_url = webhook_url
        self.callback = callback
        self.timeout_s = float(timeout_s)
        self.kind = kind
        self._lock = threading.Lock()
        # Deliveries attempted and deliveries that failed, counted
        # per channel: one alert with both a webhook and a callback
        # counts as two invocations.
        self.invoked = 0
        self.failed = 0
        self._threads = []

    @property
    def configured(self):
        """True when at least one hook channel is set."""
        return self.webhook_url is not None or self.callback is not None

    def fire(self, payload):
        """Invoke the configured hooks for one fired alert.

        ``payload`` is the alert as a plain dict (bed, raised time,
        rule, scores). Returns ``{"invoked": bool, "errors": [...]}``:
        ``invoked`` is True when at least one channel was attempted,
        ``errors`` lists synchronous failures — a callback that
        raised. A webhook is dispatched in a daemon thread, so its
        failure is not known yet at return time: it is logged and
        counted in :meth:`summary` when the thread finishes. Never
        raises.
        """
        if not self.configured:
            return {"invoked": False, "errors": []}
        invoked = False
        errors = []
        if self.callback is not None:
            invoked = True
            with self._lock:
                self.invoked += 1
            try:
                self.callback(dict(payload))
            except Exception as e:  # noqa: BLE001 - a hook must not kill a tick
                errors.append(str(e))
                with self._lock:
                    self.failed += 1
                log.warning("%s callback failed: %s", self.kind, e)
        if self.webhook_url is not None:
            invoked = True
            with self._lock:
                self.invoked += 1
            thread = threading.Thread(
                target=self._post_webhook,
                args=(dict(payload),),
                daemon=True,
            )
            with self._lock:
                self._threads.append(thread)
            thread.start()
        return {"invoked": invoked, "errors": errors}

    def _post_webhook(self, payload):
        """POST one alert payload; log and count a failure."""
        try:
            request = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                resp.read()
        except Exception as e:  # noqa: BLE001 - best effort, never fatal
            with self._lock:
                self.failed += 1
            log.warning(
                "%s webhook to %s failed: %s", self.kind, self.webhook_url, e
            )
        finally:
            with self._lock:
                self._threads = [
                    t for t in self._threads if t.is_alive()
                ]

    def wait(self, timeout=None):
        """Wait for pending webhook deliveries to finish.

        Joins the threads started so far; returns True when none are
        still running. Tests use this to assert on a stub server
        without sleeping; the ward itself never waits.
        """
        with self._lock:
            threads = list(self._threads)
        deadline = None
        if timeout is not None:
            import time

            deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        with self._lock:
            return not any(t.is_alive() for t in self._threads)

    def summary(self):
        """Hook totals as plain JSON data."""
        with self._lock:
            return {
                "configured": self.configured,
                "invoked": self.invoked,
                "failed": self.failed,
            }


# Backwards-compatible name: trend-alert hooks were the first
# event to use this machinery.
TrendAlertHooks = AlertHooks
