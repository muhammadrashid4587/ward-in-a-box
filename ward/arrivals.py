"""Walk-in arrivals: a Poisson process feeding the triage queue.

Real ED walk-ins do not arrive on a fixed schedule: they show up at
random, at a rate that rises and falls through the day. This module
models that as a Poisson process — in each tick, the number of new
arrivals is a Poisson draw with mean ``rate * dt`` — with the rate
either flat (``rate_per_hour``) or varying by hour of day
(``hourly_rates``, one rate per hour, 0-23). The draws come from the
ward's seeded RNG, so a seeded run reproduces exactly the same
arrivals. Each arrival draws an acuity and a profile from
configurable weights, then joins the triage queue like any other
arrival: it waits for a bed, or is diverted when the queue is full.
"""

import math
import random

from .vitals import PROFILES, validate_acuity

# Default acuity mix for walk-ins: most patients are mid-acuity, a
# tenth are critical — the usual ED shape.
DEFAULT_ACUITY_WEIGHTS = {1: 0.10, 2: 0.20, 3: 0.40, 4: 0.30}
# Default profile mix: half the walk-ins are stable, the rest spread
# across the other profiles.
DEFAULT_PROFILE_WEIGHTS = {
    "stable": 0.5,
    "tachycardic": 0.2,
    "hypoxic": 0.15,
    "febrile": 0.15,
}


def _poisson(rng, mean):
    """One Poisson draw with the given mean, from ``rng``.

    Knuth's algorithm while the mean is small (exact, and cheap
    there); a normal approximation, rounded and clamped at zero,
    once the mean is large enough that Knuth's loop would be slow.
    """
    if mean <= 0:
        return 0
    if mean < 30:
        limit = math.exp(-mean)
        k = 0
        p = 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1
    return max(0, int(round(rng.gauss(mean, math.sqrt(mean)))))


def _weighted_choice(rng, weights):
    """Draw one key from ``weights`` (a key -> positive weight dict)."""
    total = sum(weights.values())
    x = rng.random() * total
    for key, weight in weights.items():
        x -= weight
        if x <= 0:
            return key
    # Floating-point safety: the loop above always returns, unless
    # rounding leaves x just above zero on the last key.
    return next(reversed(list(weights)))


def _validate_weights(weights, default, kind, check_key):
    """A weights mapping with defaults kept for keys it omits.

    Every weight must be a positive number; keys are checked with
    ``check_key``. A partial mapping overrides only the keys it
    names, so a scenario can tilt one acuity without restating all
    four.
    """
    out = dict(default)
    if weights is None:
        return out
    if not isinstance(weights, dict):
        raise ValueError("%s must be an object, got %r" % (kind, weights))
    for key, value in weights.items():
        check_key(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                "%s weight for %r must be > 0, got %r" % (kind, key, value)
            )
        out[key] = float(value)
    return out


class ArrivalModel:
    """A Poisson walk-in arrival process.

    Give either ``rate_per_hour`` (one flat rate, arrivals per hour)
    or ``hourly_rates`` (a list of 24 rates, one per hour of day, so
    the ward is quiet overnight and busy at midday). ``rng`` is the
    random source — the ward passes its own seeded RNG so a seeded
    run reproduces the same arrivals; pass ``seed`` instead when
    using the model standalone.
    """

    def __init__(
        self,
        rate_per_hour=None,
        hourly_rates=None,
        acuity_weights=None,
        profile_weights=None,
        rng=None,
        seed=None,
    ):
        if rate_per_hour is not None and hourly_rates is not None:
            raise ValueError(
                "give rate_per_hour or hourly_rates, not both"
            )
        if rate_per_hour is None and hourly_rates is None:
            raise ValueError(
                "arrival model needs rate_per_hour or hourly_rates"
            )
        if rate_per_hour is not None:
            if (
                not isinstance(rate_per_hour, (int, float))
                or isinstance(rate_per_hour, bool)
                or rate_per_hour <= 0
            ):
                raise ValueError(
                    "rate_per_hour must be > 0, got %r" % (rate_per_hour,)
                )
            self.rate_per_hour = float(rate_per_hour)
            self.hourly_rates = None
        else:
            if not isinstance(hourly_rates, list) or len(hourly_rates) != 24:
                raise ValueError(
                    "hourly_rates must be a list of 24 rates, got %r"
                    % (hourly_rates,)
                )
            for hour, rate in enumerate(hourly_rates):
                if (
                    not isinstance(rate, (int, float))
                    or isinstance(rate, bool)
                    or rate < 0
                ):
                    raise ValueError(
                        "hourly_rates hour %d: rate must be >= 0, got %r"
                        % (hour, rate)
                    )
            self.hourly_rates = [float(r) for r in hourly_rates]
            self.rate_per_hour = None
        self.acuity_weights = _validate_weights(
            acuity_weights,
            DEFAULT_ACUITY_WEIGHTS,
            "acuity_weights",
            lambda key: validate_acuity(key),
        )
        self.profile_weights = _validate_weights(
            profile_weights,
            DEFAULT_PROFILE_WEIGHTS,
            "profile_weights",
            lambda key: _check_profile(key),
        )
        self.rng = rng if rng is not None else random.Random(seed)
        # Arrivals generated so far (including ones the queue diverted).
        self.generated = 0

    def rate_at(self, hour):
        """Arrivals per hour at ``hour`` (0-23, fractional hours ok)."""
        if self.hourly_rates is not None:
            return self.hourly_rates[int(hour) % 24]
        return self.rate_per_hour

    def generate(self, now, dt, hour):
        """Arrivals for one tick: a Poisson count over ``dt`` seconds.

        Returns a list of ``{"arrival_t", "acuity", "profile"}`` dicts
        — the same shape a scheduled arrival has once it joins the
        queue. ``now`` is the tick's elapsed time, ``hour`` the ward
        clock hour (0-23) the rate is read at.
        """
        mean = self.rate_at(hour) * dt / 3600.0
        count = _poisson(self.rng, mean)
        arrivals = []
        for _ in range(count):
            arrivals.append(
                {
                    "arrival_t": round(now, 3),
                    "acuity": _weighted_choice(self.rng, self.acuity_weights),
                    "profile": _weighted_choice(self.rng, self.profile_weights),
                }
            )
        self.generated += count
        return arrivals


def _check_profile(profile):
    if profile not in PROFILES:
        raise ValueError("unknown profile: %r" % (profile,))
