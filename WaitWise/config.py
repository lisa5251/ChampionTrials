"""Shared settings. Add a service or a counter here and the whole app picks it up."""

SERVICES = [
    "General Inquiry",
    "Payments",
    "Technical Support",
]

SERVICE_INFO = {
    "General Inquiry": "Information and general assistance",
    "Payments": "Billing and payment support",
    "Technical Support": "Help with technical issues",
}

COUNTERS = [f"Counter {i}" for i in range(1, 5)]