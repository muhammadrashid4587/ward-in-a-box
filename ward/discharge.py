"""Discharge planning: planned discharges with a lead time.

A discharge order is placed for a bed — by a scenario, or live with
:func:`DischargePlanner.place_order` — and the patient is discharged
``lead_time_s`` seconds later, when the order's expected time arrives.
The bed then goes through cleaning as usual, so a planned discharge
followed by a triage admission tests exactly the gap a real ward has
between "the doctor says they can go home" and the bed being free.

If cleaning is backed up — the ward's ``cleaning_capacity`` is reached
— the discharge is delayed: the patient stays in the bed, still
monitored, until a cleaning slot frees up, and the delay is recorded
on the order. An order for a bed whose patient already left (a chaos
``discharge`` event fired first, say) is cancelled, not forced.

An auto order is the ward's own call, not a doctor's, so it is
withdrawn when the reason for it goes away: a live auto order is
cancelled the first tick its bed stops being discharge-ready — an
alarm restarted the stability clock, or a dose became outstanding
— while manual and scheduled orders stand, because a doctor's
order is not the ward's to withdraw. A bed whose auto order was
cancelled gets a fresh one when it flags ready again, through the
normal one-live-order rule. Every cancellation is stamped on the
order (``cancelled_t``, ``cancel_reason``) and counted in the
summary (``auto_cancelled`` counts the auto ones).

Paperwork timing: the planner also measures how long the discharge
paperwork takes, from the moment a bed is first flagged
discharge-ready — ``ready_t`` on the order — to the order being
placed (``ready_to_order_s``) and to the patient actually leaving
(``ready_to_discharge_s``). An order placed before the bed was ready
has no ready-to-order time; an order for a bed that never flags ready
has no timing at all. The summary aggregates the measured orders
(mean and max), so a ward can see whether the paperwork or the
cleaning roster is what holds a bed up.
"""

# Default lead time between a discharge order and the discharge:
# an hour, the same shape as a real ward's morning rounds to
# afternoon discharge.
DEFAULT_LEAD_TIME_S = 3600.0


class DischargePlanner:
    """Planned discharges for one ward.

    ``lead_time_s`` is the default delay between an order and the
    discharge; an order may override it. ``orders`` are scheduled
    orders, ``{"bed": n, "at": t}`` (with optional ``"lead_time_s"``),
    placed when their ``at`` time arrives. Orders placed live with
    :meth:`place_order` take effect at once.
    """

    def __init__(self, lead_time_s=DEFAULT_LEAD_TIME_S, orders=None):
        if (
            not isinstance(lead_time_s, (int, float))
            or isinstance(lead_time_s, bool)
            or lead_time_s <= 0
        ):
            raise ValueError(
                "discharge lead_time_s must be > 0, got %r" % (lead_time_s,)
            )
        self.lead_time_s = float(lead_time_s)
        # Scheduled orders not yet placed, sorted by time.
        self.scheduled_orders = []
        for i, o in enumerate(orders or []):
            if not isinstance(o, dict):
                raise ValueError("discharge order %d: must be an object" % i)
            try:
                bed = o["bed"]
            except KeyError:
                raise ValueError("discharge order %d: missing 'bed'" % i)
            if not isinstance(bed, int) or isinstance(bed, bool) or bed < 0:
                raise ValueError(
                    "discharge order %d: bed must be an int >= 0, got %r"
                    % (i, bed)
                )
            try:
                at = float(o["at"])
            except KeyError:
                raise ValueError("discharge order %d: missing 'at'" % i)
            if at < 0:
                raise ValueError(
                    "discharge order %d: 'at' must be >= 0" % i
                )
            lead = o.get("lead_time_s", self.lead_time_s)
            if (
                not isinstance(lead, (int, float))
                or isinstance(lead, bool)
                or lead <= 0
            ):
                raise ValueError(
                    "discharge order %d: lead_time_s must be > 0, got %r"
                    % (i, lead)
                )
            self.scheduled_orders.append(
                {"bed": bed, "at": round(at, 3), "lead_time_s": float(lead)}
            )
        self.scheduled_orders.sort(key=lambda o: o["at"])
        self._next_scheduled = 0
        # When each bed was first flagged discharge-ready in its
        # current stay, ``{bed: t}`` — the start of the paperwork
        # clock. The ward notes it (see :meth:`note_ready`) and
        # clears it when the bed stops being ready or leaves.
        self._ready_since = {}
        # Placed orders: {"bed", "ordered_t", "expected_t", "status",
        # "actual_t", "delay_s", "ready_t", "ready_to_order_s",
        # "ready_to_discharge_s", "cancelled_t", "cancel_reason"} —
        # status is "planned" until the expected time, then
        # "discharged", "delayed" (cleaning backed up, still
        # waiting), or "cancelled" (patient already gone, or — for
        # an auto order — the bed stopped being discharge-ready).
        # "cancelled_t" is when a cancelled order was withdrawn
        # and "cancel_reason" why ("patient_left" or "not_ready");
        # both are None for orders that were never cancelled.
        self.orders = []

    def place_order(self, bed, ordered_at, lead_time_s=None, source="manual"):
        """Place a discharge order for ``bed`` at ``ordered_at``.

        The discharge is expected at ``ordered_at + lead_time_s``
        (the planner default when omitted). ``source`` says where
        the order came from: ``"manual"`` (placed live), ``"scheduled"``
        (from the scenario), or ``"auto"`` (raised by the ward when
        the bed was flagged discharge-ready). Returns the order.
        """
        if not isinstance(bed, int) or isinstance(bed, bool) or bed < 0:
            raise ValueError("bed must be an int >= 0, got %r" % (bed,))
        if source not in ("manual", "scheduled", "auto"):
            raise ValueError(
                "source must be 'manual', 'scheduled', or 'auto', got %r"
                % (source,)
            )
        lead = self.lead_time_s if lead_time_s is None else lead_time_s
        if (
            not isinstance(lead, (int, float))
            or isinstance(lead, bool)
            or lead <= 0
        ):
            raise ValueError("lead_time_s must be > 0, got %r" % (lead,))
        ready_t = self._ready_since.get(bed)
        order = {
            "bed": bed,
            "ordered_t": round(ordered_at, 3),
            "expected_t": round(ordered_at + float(lead), 3),
            "status": "planned",
            "actual_t": None,
            "delay_s": None,
            "source": source,
            # When the bed was first flagged discharge-ready, if it
            # was ready when (or before) the order was placed; the
            # paperwork timings are filled in when the patient
            # leaves (see :meth:`_finish_timing`).
            "ready_t": ready_t,
            "ready_to_order_s": (
                round(ordered_at - ready_t, 3)
                if ready_t is not None and ready_t <= ordered_at
                else None
            ),
            "ready_to_discharge_s": None,
            # When a cancelled order was withdrawn and why; None
            # for orders that were never cancelled.
            "cancelled_t": None,
            "cancel_reason": None,
        }
        self.orders.append(order)
        return order

    def note_ready(self, bed, now):
        """Note that ``bed`` was flagged discharge-ready at ``now``.

        The first flagging in a stay starts the paperwork clock;
        a bed already flagged keeps its first time. A live order
        for the bed that was placed before the bed was ready picks
        the time up too, so its ready-to-discharge time can still
        be measured.
        """
        now = round(now, 3)
        if bed not in self._ready_since:
            self._ready_since[bed] = now
        for order in self.orders:
            if (
                order["bed"] == bed
                and order["status"] in ("planned", "delayed")
                and order["ready_t"] is None
            ):
                order["ready_t"] = now

    def note_not_ready(self, bed):
        """Clear ``bed``'s paperwork clock: it stopped being ready,
        or the patient left. An order already discharged keeps the
        ``ready_t`` it recorded; a live order keeps its own too —
        the clock restarts only for a fresh flagging, when a new
        order is placed."""
        self._ready_since.pop(bed, None)

    def has_live_order(self, bed):
        """True when ``bed`` already has a discharge order in flight:
        one scheduled but not yet placed, or one placed and still
        planned or delayed. A discharged or cancelled order does not
        count, so a new patient admitted to the bed later can get a
        fresh order of their own."""
        if any(
            o["bed"] == bed
            for o in self.scheduled_orders[self._next_scheduled :]
        ):
            return True
        return any(
            o["bed"] == bed and o["status"] in ("planned", "delayed")
            for o in self.orders
        )

    def has_auto_order(self, bed):
        """True when ``bed`` has a live auto-raised order: one the
        ward placed when the bed was flagged discharge-ready and
        that has not discharged or been cancelled yet."""
        return any(
            o["bed"] == bed
            and o.get("source") == "auto"
            and o["status"] in ("planned", "delayed")
            for o in self.orders
        )

    def cancel_auto_orders(self, bed, now):
        """Cancel ``bed``'s live auto orders at ``now``.

        Only orders the ward raised itself — ``source`` "auto" —
        are withdrawn, and only while they are still live
        ("planned" or "delayed"). Manual and scheduled orders
        are a doctor's order, not the ward's to withdraw, so
        they are left alone. Each cancelled order is stamped
        with when it was withdrawn and why (``"not_ready"``:
        the bed stopped being discharge-ready). Returns the
        orders cancelled, so the caller can see what changed.
        """
        now = round(now, 3)
        cancelled = []
        for order in self.orders:
            if (
                order["bed"] == bed
                and order.get("source") == "auto"
                and order["status"] in ("planned", "delayed")
            ):
                order["status"] = "cancelled"
                order["cancelled_t"] = now
                order["cancel_reason"] = "not_ready"
                cancelled.append(order)
        return cancelled

    def tick(self, now, ward):
        """Place due scheduled orders and execute due ones.

        An order whose expected time has arrived is discharged at
        once when a cleaning slot is free; when cleaning is backed
        up it stays "delayed" and the ward completes it when a slot
        frees up (see :meth:`complete`). An order for a bed whose
        patient already left is cancelled.
        """
        now = round(now, 3)
        while self._next_scheduled < len(self.scheduled_orders):
            order = self.scheduled_orders[self._next_scheduled]
            if order["at"] > now:
                break
            self._next_scheduled += 1
            self.place_order(
                order["bed"], order["at"], order["lead_time_s"],
                source="scheduled",
            )
        for order in self.orders:
            if order["status"] != "planned" or order["expected_t"] > now:
                continue
            result = ward.discharge_planned(order["bed"], order["expected_t"])
            if result == "discharged":
                order["status"] = "discharged"
                order["actual_t"] = now
                order["delay_s"] = 0.0
                self._finish_timing(order, now)
            elif result == "queued":
                order["status"] = "delayed"
            else:
                order["status"] = "cancelled"
                order["cancelled_t"] = now
                order["cancel_reason"] = "patient_left"

    def complete(self, bed, now):
        """Mark a delayed order for ``bed`` discharged at ``now``.

        Called by the ward when a discharge that was queued behind
        a full cleaning roster finally starts cleaning. The delay
        is how long past the expected time the discharge landed.
        """
        now = round(now, 3)
        for order in self.orders:
            if order["bed"] == bed and order["status"] == "delayed":
                order["status"] = "discharged"
                order["actual_t"] = now
                order["delay_s"] = round(now - order["expected_t"], 3)
                self._finish_timing(order, now)
                return order
        return None

    def _finish_timing(self, order, actual_t):
        """Fill in an order's paperwork timings at discharge.

        ``ready_to_discharge_s`` is the time from the bed first
        being flagged ready to the patient leaving — so it includes
        any cleaning delay. ``ready_to_order_s`` is only meaningful
        when the bed was already ready when the order was placed;
        an order placed before the bed was ready has none. An order
        for a bed that never flagged ready has neither.
        """
        ready_t = order.get("ready_t")
        if ready_t is None:
            order["ready_to_order_s"] = None
            order["ready_to_discharge_s"] = None
            return
        if ready_t <= order["ordered_t"]:
            order["ready_to_order_s"] = round(
                order["ordered_t"] - ready_t, 3
            )
        order["ready_to_discharge_s"] = round(actual_t - ready_t, 3)

    def summary(self, now):
        """Placed orders and discharge totals as plain JSON data."""
        now = round(now, 3)
        discharged = [o for o in self.orders if o["status"] == "discharged"]
        delayed = [o for o in discharged if (o["delay_s"] or 0) > 0]
        total_delay = round(sum(o["delay_s"] or 0 for o in discharged), 3)
        # Paperwork timing, over discharged orders whose bed flagged
        # ready: how long from ready to the order and from ready to
        # leaving. A ready-to-order time exists only when the bed was
        # already ready when the order was placed.
        measured = [
            o for o in discharged if o.get("ready_t") is not None
        ]
        to_order = [
            o["ready_to_order_s"]
            for o in measured
            if o.get("ready_to_order_s") is not None
        ]
        to_discharge = [
            o["ready_to_discharge_s"]
            for o in measured
            if o.get("ready_to_discharge_s") is not None
        ]
        return {
            "t": now,
            "lead_time_s": self.lead_time_s,
            "orders": [dict(o) for o in self.orders],
            "planned": len(self.orders),
            "auto_planned": sum(
                1 for o in self.orders if o.get("source") == "auto"
            ),
            "discharged": len(discharged),
            "delayed": len(delayed),
            "cancelled": sum(
                1 for o in self.orders if o["status"] == "cancelled"
            ),
            "auto_cancelled": sum(
                1
                for o in self.orders
                if o["status"] == "cancelled"
                and o.get("source") == "auto"
            ),
            "pending": sum(
                1
                for o in self.orders
                if o["status"] in ("planned", "delayed")
            ),
            "total_delay_s": total_delay,
            "mean_delay_s": (
                round(total_delay / len(delayed), 3) if delayed else 0.0
            ),
            "timing": {
                "measured": len(measured),
                "mean_ready_to_order_s": (
                    round(sum(to_order) / len(to_order), 3)
                    if to_order
                    else 0.0
                ),
                "mean_ready_to_discharge_s": (
                    round(sum(to_discharge) / len(to_discharge), 3)
                    if to_discharge
                    else 0.0
                ),
                "max_ready_to_discharge_s": (
                    round(max(to_discharge), 3) if to_discharge else 0.0
                ),
            },
        }
