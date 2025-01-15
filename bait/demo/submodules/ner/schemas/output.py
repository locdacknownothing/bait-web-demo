from typing import Optional
from dataclasses import dataclass


@dataclass
class NERResult:
    address: Optional[list[str]] = None
    keyword: Optional[list[str]] = None
    name: Optional[list[str]] = None
