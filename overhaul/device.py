from __future__ import annotations


class Device():
    def __init__(
            self, 
            dname: str = "CPU"
        ) -> Device:
        # Verify that the device is supported by the library.
        if dname not in ["CPU"]:
            raise ValueError(f"The device you specified, '{dname},' is not supported.")

        self.dname = dname

    # Standardizes the format of the device for comparison/viewing.
    def standard(self) -> str:
        return self.dname