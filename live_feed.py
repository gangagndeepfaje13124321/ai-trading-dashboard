import math


def _format_error(provider, exc):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    reason = getattr(response, "reason", None)
    if status is not None:
        detail = f"HTTP {status}" + (f" {reason}" if reason else "")
    else:
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
    return f"{provider}: {detail}"


def get_first_valid_price(providers):
    """Return the first finite, positive price from ordered provider callables."""
    errors = []
    for name, fetch_price in providers:
        try:
            price = float(fetch_price())
            if not math.isfinite(price) or price <= 0:
                raise ValueError("provider returned a non-positive or non-finite price")
            return price, name, errors
        except Exception as exc:
            errors.append(_format_error(name, exc))

    details = "; ".join(errors) if errors else "no providers configured"
    raise RuntimeError(f"Both live providers failed ({details}).")
