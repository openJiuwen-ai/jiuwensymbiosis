# Synthetic temperature controller protocol

This document describes a local software fixture, not a vendor device or a physical safety range.

## Commands

GET /status returns {"status":"IDLE"} for the idle fixture.
GET /temperature returns {"temperature_c":20.0}.
POST /setpoint accepts {"temperature_c":N}, where N is a number from 0 through 80; it returns {"accepted":true}.
The accepted response confirms the protocol request only; it does not mean a target temperature was reached.
OPAQUE is a deliberately undocumented binary command and remains blocked.
