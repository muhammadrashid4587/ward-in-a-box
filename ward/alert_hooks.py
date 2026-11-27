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

Seven events use hooks. A trend alert (a NEWS2 score rising fast
enough to trip the jump or creep rule) fires the ward's
``trend_alert_hooks``; a deterioration episode opening (a
patient's NEWS2 risk leaving low) fires the ward's
``escalation_hooks``; a boarding-full alarm episode raising (a
patient diverted while ED boarding is full) fires the ward's
``boarding_full_hooks``; a wait-time alarm raising (a waiting
patient passing their acuity's wait threshold) fires the ward's
``wait_time_hooks``; a dose missed — given up on because the ward
was too thin to give it inside its grace window, or skipped
because the pharmacy was out of that medication — fires the
ward's ``missed_dose_hooks``; a discharge-readiness flip (a
patient becoming discharge-ready, or stopping being ready)
fires the ward's ``discharge_readiness_hooks``; a change in the
discharge-priority order (a bed becoming ready, a ready bed
leaving, or the ranking reordering) fires the ward's
``discharge_priority_hooks``; an oxygen outage starting, ending,
or pushing a patient onto the SpO2 floor fires the ward's
``oxygen_outage_hooks``; a power outage starting or ending fires
the ward's ``power_outage_hooks``; a sensor-drift event starting
or ending fires the ward's ``sensor_drift_hooks``; a bed-blocker
event starting or ending fires the ward's
``bed_blocker_hooks``; a network outage starting or ending fires
the ward's ``network_outage_hooks``; a medication stockout
starting or ending fires the ward's
``medication_stockout_hooks``; an infusion pump failing or
restarting fires the ward's ``pump_failure_hooks``; a lab
running slow starting or returning to normal fires the ward's
``lab_delay_hooks``; a lab result going overdue — past its
due time and still not back — or landing after being
overdue fires the ward's ``lab_overdue_hooks``.
All sixteen share
the same ``AlertHooks`` machinery — the same webhook and
callback channels, the same best-effort delivery — just a
different event payload.

A fired event records whether the hook was invoked (``"hook_invoked"``
on the alert or episode, False when no hook is configured) and any
synchronous error (``"hook_errors"``). The early-warning summary
carries the ward totals — configured, invoked, failed — so a run
summary can say how many alerts reached the outside world. Hooks
never touch the vital-sign alarm list, and a ward with no hooks
configured behaves exactly as before.

A failed webhook POST is retried with exponential backoff, up to
``max_retries`` retries (2 by default, so 3 attempts in total),
in the same daemon thread — a slow or dead endpoint still never
blocks the ward tick. A 2xx response counts as delivered; a
connection error, a timeout, or any other status counts as a
failed attempt. Each fired event records how many retries its
delivery took (``"hook_retries"``) and the final outcome
(``"hook_final_status"``: ``"delivered"``, ``"failed"``,
``"pending"`` while a delivery is still in flight, or
``"not_invoked"`` when no hook is configured), and the ward
totals carry the retry count and how many deliveries eventually
landed. A callback that raises is not retried — a callback is
in-process code, so a failure there is a bug, not a transient
network fault — and it is counted as failed at once.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger("ward.alert_hooks")

# Seconds a webhook POST may take before it is abandoned. The POST
# runs in a daemon thread, so this never stalls the ward — it only
# bounds how long the delivery thread lingers.
DEFAULT_WEBHOOK_TIMEOUT_S = 5.0

# How many times a failed webhook POST is retried, and the base
# delay before the first retry. The delay doubles after each
# failed attempt (base, 2x base, 4x base, ...), so the defaults
# retry at 1 s and 2 s after the first failure.
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 1.0


def _validate_hooks(value, kind, label):
    """Normalize a hook config for one event kind.

    None means no hooks (off); an ``AlertHooks`` passes through;
    a dict may set ``webhook_url`` (an http(s) URL to POST the
    alert payload to), ``callback`` (an in-process callable,
    Python API only — a scenario file cannot hold one),
    ``timeout_s`` (a number above 0, how long a webhook POST may
    take), ``max_retries`` (an integer 0 or above, how many times
    a failed webhook POST is retried), and ``retry_backoff_s``
    (a number above 0, the delay before the first retry, doubling
    after each failed attempt). Returns an ``AlertHooks``.
    """
    if value is None:
        return AlertHooks(kind=kind)
    if isinstance(value, AlertHooks):
        return value
    if not isinstance(value, dict):
        raise ValueError("%s must be a dict, got %r" % (label, value))
    unknown = sorted(
        set(value)
        - {
            "webhook_url",
            "callback",
            "timeout_s",
            "max_retries",
            "retry_backoff_s",
        }
    )
    if unknown:
        raise ValueError(
            "%s has unknown keys: %s" % (label, ", ".join(unknown))
        )
    return AlertHooks(
        webhook_url=value.get("webhook_url"),
        callback=value.get("callback"),
        timeout_s=value.get("timeout_s", DEFAULT_WEBHOOK_TIMEOUT_S),
        max_retries=value.get("max_retries", DEFAULT_MAX_RETRIES),
        retry_backoff_s=value.get(
            "retry_backoff_s", DEFAULT_RETRY_BACKOFF_S
        ),
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


def validate_boarding_full_hooks(value):
    """Normalize a boarding-full hook config (see :func:`_validate_hooks`).

    Fires when a boarding-full alarm episode raises — a patient
    diverted while ED boarding is full — rather than when a NEWS2
    event fires.
    """
    return _validate_hooks(value, "boarding-full", "boarding_full_hooks")


def validate_wait_time_hooks(value):
    """Normalize a wait-time hook config (see :func:`_validate_hooks`).

    Fires when a wait-time alarm raises — a waiting patient
    passing their acuity's wait threshold — rather than when a
    NEWS2 event fires.
    """
    return _validate_hooks(value, "wait-time", "wait_time_hooks")


def validate_missed_dose_hooks(value):
    """Normalize a missed-dose hook config (see :func:`_validate_hooks`).

    Fires when a dose is missed — its grace window expired before
    the ward could give it, or it was skipped because the pharmacy
    was out of that medication — rather than when a NEWS2 event
    fires.
    """
    return _validate_hooks(value, "missed-dose", "missed_dose_hooks")


def validate_discharge_readiness_hooks(value):
    """Normalize a discharge-readiness hook config (see
    :func:`_validate_hooks`).

    Fires when a patient's discharge-readiness flag flips — to
    ready or back to not-ready — rather than when a NEWS2 event
    fires.
    """
    return _validate_hooks(
        value, "discharge-readiness", "discharge_readiness_hooks"
    )


def validate_discharge_priority_hooks(value):
    """Normalize a discharge-priority hook config (see
    :func:`_validate_hooks`).

    Fires when the discharge-priority order changes — a bed
    becoming ready, a ready bed leaving, or the ranking
    reordering — rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "discharge-priority", "discharge_priority_hooks"
    )


def validate_oxygen_outage_hooks(value):
    """Normalize an oxygen-outage hook config (see
    :func:`_validate_hooks`).

    Fires when an oxygen outage starts or ends, and when a
    patient on oxygen first hits the outage's SpO2 floor —
    rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "oxygen-outage", "oxygen_outage_hooks"
    )


def validate_power_outage_hooks(value):
    """Normalize a power-outage hook config (see
    :func:`_validate_hooks`).

    Fires when a power outage starts or ends — when the mains
    fails and when it returns — rather than when a NEWS2 event
    fires.
    """
    return _validate_hooks(
        value, "power-outage", "power_outage_hooks"
    )


def validate_sensor_drift_hooks(value):
    """Normalize a sensor-drift hook config (see
    :func:`_validate_hooks`).

    Fires when a sensor-drift event starts or ends — when a
    bed's sensor starts miscalibrating and when the drift
    window closes — rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "sensor-drift", "sensor_drift_hooks"
    )


def validate_bed_blocker_hooks(value):
    """Normalize a bed-blocker hook config (see
    :func:`_validate_hooks`).

    Fires when a bed-blocker event starts or ends — when a
    bed goes out of use and when it comes back — rather than
    when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "bed-blocker", "bed_blocker_hooks"
    )


def validate_network_outage_hooks(value):
    """Normalize a network-outage hook config (see
    :func:`_validate_hooks`).

    Fires when a network outage starts or ends — when the
    link to the central station fails and when it returns —
    rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "network-outage", "network_outage_hooks"
    )


def validate_medication_stockout_hooks(value):
    """Normalize a medication-stockout hook config (see
    :func:`_validate_hooks`).

    Fires when a medication stockout starts or ends — when
    the pharmacy runs out of a medication and when the stock
    returns — rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "medication-stockout", "medication_stockout_hooks"
    )


def validate_pump_failure_hooks(value):
    """Normalize a pump-failure hook config (see
    :func:`_validate_hooks`).

    Fires when an infusion pump fails or restarts — when
    a bedside pump stops delivering its infusion and when
    it is running again — rather than when a NEWS2 event
    fires.
    """
    return _validate_hooks(
        value, "pump-failure", "pump_failure_hooks"
    )


def validate_lab_delay_hooks(value):
    """Normalize a lab-delay hook config (see
    :func:`_validate_hooks`).

    Fires when a lab delay starts or ends — when the lab
    starts running slow and when it is back to normal —
    rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "lab-delay", "lab_delay_hooks"
    )


def validate_lab_overdue_hooks(value):
    """Normalize a lab-overdue hook config (see
    :func:`_validate_hooks`).

    Fires when a lab result goes overdue — past its due
    time and still not back — and when it lands after
    being overdue — rather than when a NEWS2 event fires.
    """
    return _validate_hooks(
        value, "lab-overdue", "lab_overdue_hooks"
    )


class AlertHooks:
    """The configured hooks for one ward event, shared by its monitors.

    ``kind`` labels the event in log messages (``"trend-alert"``,
    ``"escalation"``, ``"boarding-full"``, ``"wait-time"``,
    ``"missed-dose"``, ``"discharge-readiness"``,
    ``"discharge-priority"``, ``"oxygen-outage"``,
    ``"power-outage"``, ``"sensor-drift"``, ``"bed-blocker"``, or
    ``"network-outage"``, ``"medication-stockout"``,
    ``"pump-failure"``, ``"lab-delay"``, or ``"lab-overdue"``); it changes
    nothing about delivery.
    """

    def __init__(self, webhook_url=None, callback=None,
                 timeout_s=DEFAULT_WEBHOOK_TIMEOUT_S,
                 max_retries=DEFAULT_MAX_RETRIES,
                 retry_backoff_s=DEFAULT_RETRY_BACKOFF_S,
                 kind="trend-alert"):
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
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError(
                "alert hooks max_retries must be an integer >= 0, "
                "got %r" % (max_retries,)
            )
        if (
            isinstance(retry_backoff_s, bool)
            or not isinstance(retry_backoff_s, (int, float))
            or retry_backoff_s <= 0
        ):
            raise ValueError(
                "alert hooks retry_backoff_s must be a number > 0, "
                "got %r" % (retry_backoff_s,)
            )
        self.webhook_url = webhook_url
        self.callback = callback
        self.timeout_s = float(timeout_s)
        self.max_retries = max_retries
        self.retry_backoff_s = float(retry_backoff_s)
        self.kind = kind
        self._lock = threading.Lock()
        # Deliveries attempted and deliveries that failed, counted
        # per channel: one alert with both a webhook and a callback
        # counts as two invocations. ``failed`` counts channels
        # that never delivered after all their retries; ``retries``
        # counts every re-attempt; ``delivered`` counts channels
        # that eventually landed, including after a retry.
        self.invoked = 0
        self.failed = 0
        self.retries = 0
        self.delivered = 0
        self._threads = []

    @property
    def configured(self):
        """True when at least one hook channel is set."""
        return self.webhook_url is not None or self.callback is not None

    def fire(self, payload, record=None):
        """Invoke the configured hooks for one fired alert.

        ``payload`` is the alert as a plain dict (bed, raised time,
        rule, scores). ``record`` is the alert or episode dict the
        delivery outcome is recorded on, when given: it gets
        ``"hook_retries"`` (retry attempts so far for this event)
        and ``"hook_final_status"`` (``"delivered"`` when every
        invoked channel landed, ``"failed"`` when any channel
        gave up, ``"pending"`` while a webhook delivery is still
        in flight, ``"not_invoked"`` when no hook is configured).
        The record is updated in place as the delivery settles,
        so a snapshot taken later shows the final outcome.
        Returns ``{"invoked": bool, "errors": [...], "retries": int,
        "final_status": str}``: ``invoked`` is True when at least
        one channel was attempted, ``errors`` lists synchronous
        failures — a callback that raised. A webhook is dispatched
        in a daemon thread, so its failure is not known yet at
        return time: it is logged and counted in :meth:`summary`
        when the thread finishes, and the record is updated then.
        Never raises.
        """
        if not self.configured:
            if record is not None:
                record["hook_retries"] = 0
                record["hook_final_status"] = "not_invoked"
            return {
                "invoked": False,
                "errors": [],
                "retries": 0,
                "final_status": "not_invoked",
            }
        # Per-delivery state, shared with the webhook thread:
        # each channel's outcome so far, and the retries spent.
        # ``webhook``/``callback`` are None (not configured),
        # "pending" (in flight), "delivered", or "failed".
        state = {
            "record": record,
            "webhook": None,
            "callback": None,
            "retries": 0,
        }
        invoked = False
        errors = []
        if self.callback is not None:
            invoked = True
            with self._lock:
                self.invoked += 1
            try:
                self.callback(dict(payload))
                state["callback"] = "delivered"
                with self._lock:
                    self.delivered += 1
            except Exception as e:  # noqa: BLE001 - a hook must not kill a tick
                errors.append(str(e))
                state["callback"] = "failed"
                with self._lock:
                    self.failed += 1
                log.warning("%s callback failed: %s", self.kind, e)
        if self.webhook_url is not None:
            invoked = True
            with self._lock:
                self.invoked += 1
            state["webhook"] = "pending"
            thread = threading.Thread(
                target=self._post_webhook,
                args=(dict(payload), state),
                daemon=True,
            )
            with self._lock:
                self._threads.append(thread)
            thread.start()
        final = self._settle(state)
        return {
            "invoked": invoked,
            "errors": errors,
            "retries": state["retries"],
            "final_status": final,
        }

    def _settle(self, state):
        """Recompute one delivery's final status and record it.

        ``"pending"`` while any channel is still in flight,
        ``"failed"`` when any channel gave up, ``"delivered"``
        when every invoked channel landed. Writes
        ``hook_retries`` and ``hook_final_status`` onto the
        delivery's record, when there is one. Returns the status.
        """
        with self._lock:
            statuses = [
                s
                for s in (state["webhook"], state["callback"])
                if s is not None
            ]
            if "pending" in statuses:
                final = "pending"
            elif "failed" in statuses:
                final = "failed"
            elif statuses:
                final = "delivered"
            else:
                final = "not_invoked"
            record = state["record"]
            if record is not None:
                record["hook_retries"] = state["retries"]
                record["hook_final_status"] = final
            return final

    def _post_webhook(self, payload, state):
        """POST one alert payload, retrying failures with backoff.

        A 2xx response counts as delivered; a connection error, a
        timeout, or any other status counts as a failed attempt,
        retried after ``retry_backoff_s`` seconds, doubling after
        each failed attempt, up to ``max_retries`` retries. Runs
        in a daemon thread, so the backoff sleeps never stall a
        ward tick. Each retry and the final outcome are counted
        in :meth:`summary` and recorded on the delivery's record
        via :meth:`_settle`.
        """
        try:
            self._post_webhook_with_retries(payload, state)
        finally:
            with self._lock:
                self._threads = [
                    t for t in self._threads if t.is_alive()
                ]

    def _post_webhook_with_retries(self, payload, state):
        """The retry loop for one webhook delivery (see
        :meth:`_post_webhook`). Returns when the payload is
        delivered or the retries are exhausted."""
        attempts = 0
        while True:
            attempts += 1
            error = None
            try:
                request = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as resp:
                    resp.read()
                    if 200 <= resp.status < 300:
                        with self._lock:
                            self.delivered += 1
                        state["webhook"] = "delivered"
                        self._settle(state)
                        return
                    error = "webhook returned HTTP %d" % resp.status
            except Exception as e:  # noqa: BLE001 - best effort, never fatal
                error = str(e)
            if attempts > self.max_retries:
                with self._lock:
                    self.failed += 1
                log.warning(
                    "%s webhook to %s failed after %d attempts: %s",
                    self.kind,
                    self.webhook_url,
                    attempts,
                    error,
                )
                state["webhook"] = "failed"
                self._settle(state)
                return
            with self._lock:
                self.retries += 1
                state["retries"] += 1
            delay = self.retry_backoff_s * (2 ** (attempts - 1))
            log.warning(
                "%s webhook to %s failed (%s); retry %d/%d in %.1fs",
                self.kind,
                self.webhook_url,
                error,
                attempts,
                self.max_retries,
                delay,
            )
            time.sleep(delay)
        # Unreachable: the loop returns on delivery or on giving up.


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
                "retries": self.retries,
                "delivered": self.delivered,
            }


# Backwards-compatible name: trend-alert hooks were the first
# event to use this machinery.
TrendAlertHooks = AlertHooks
