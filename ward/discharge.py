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
        # Placed orders: {"bed", "ordered_t", "expected_t", "status",
        # "actual_t", "delay_s"} — status is "planned" until the
        # expected time, then "discharged", "delayed" (cleaning backed
        # up, still waiting), or "cancelled" (patient already gone).
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
        order = {
            "bed": bed,
            "ordered_t": round(ordered_at, 3),
            "expected_t": round(ordered_at + float(lead), 3),
            "status": "planned",
            "actual_t": None,
            "delay_s": None,
            "source": source,
        }
        self.orders.append(order)
        return order

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
            elif result == "queued":
                order["status"] = "delayed"
            else:
                order["status"] = "cancelled"

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
                return order
        return None

    def summary(self, now):
        """Placed orders and discharge totals as plain JSON data."""
        now = round(now, 3)
        discharged = [o for o in self.orders if o["status"] == "discharged"]
        delayed = [o for o in discharged if (o["delay_s"] or 0) > 0]
        total_delay = round(sum(o["delay_s"] or 0 for o in discharged), 3)
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
            "pending": sum(
                1
                for o in self.orders
                if o["status"] in ("planned", "delayed")
            ),
            "total_delay_s": total_delay,
            "mean_delay_s": (
                round(total_delay / len(delayed), 3) if delayed else 0.0
            ),
        }
