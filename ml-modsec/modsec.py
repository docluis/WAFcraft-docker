# This file is a modified version of modsecurity-cli/main.py
# https://github.com/AvalZ/modsecurity-cli
# TODO: proper attribution

import base64
from ModSecurity import ModSecurity # type: ignore
from ModSecurity import RulesSet # type: ignore
from ModSecurity import Transaction # type: ignore
from ModSecurity import LogProperty # type: ignore

import re
import glob
from urllib.parse import urlparse, urlencode
from typing import List, Optional
from typing_extensions import Annotated
from enum import Enum


class Severity(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, severity_id, score):
        self.id = severity_id
        self.score = score

    EMERGENCY = 0, 0  # not used in CRS
    ALERT = 1, 0  # not used in CRS
    CRITICAL = 2, 5
    ERROR = 3, 4
    WARNING = 4, 3
    NOTICE = 5, 2
    INFO = 6, 0  # not used in CRS
    DEBUG = 7, 0  # not used in CRS


def get_paranoia_level(rule):
    return next(
        (int(tag.split("/")[1]) for tag in rule.m_tags if "paranoia-level" in tag), 1
    )


def version(value: bool):
    if value:
        modsec = ModSecurity()
        print(modsec.whoAmI())
        exit()


def init_modsec():
    modsec = ModSecurity()
    return modsec


def get_activated_rules(
    payloads_base64: List[str],
    keys: Optional[List[str]] = None,
    request_body: Optional[str] = None,  # Changed to string type for simplicity
    base_uri: str = "http://www.modsecurity.org/test",
    method: str = "",
    headers: Optional[List[str]] = [],
    paranoia_level: int = 1,
    configs: Optional[List[str]] = [
        "/app/modsecurity-cli/conf/modsecurity.conf",
        "/app/modsecurity-cli/conf/crs-setup.conf",
    ],
    rules_path: str = "rules",
    verbose: bool = False,
    version: Optional[bool] = None,
    logs: bool = False,
    modsec: ModSecurity = init_modsec(),
):
    if not logs:
        # disable ModSecurity callback logs
        modsec.setServerLogCb2(lambda x, y: None, LogProperty.RuleMessageLogProperty)

    if not method:
        method = "POST" if request_body else "GET"

    if not keys:
        keys = ["q"]

    # base64 decode payloads
    payloads = [base64.b64decode(payload).decode("utf-8") for payload in payloads_base64]

    encoded_query = urlencode(dict(zip(keys, payloads)))
    full_url = f"{base_uri}?{encoded_query}"
    parsed_url = urlparse(full_url)

    rules = RulesSet()

    # Load basic conf
    for config in configs:
        rules.loadFromUri(config)

    # Load rules
    for rule_path in sorted(glob.glob(f"{rules_path}/*.conf")):
        # Unsorted rules cause unexpcted behaviors for SETVAR
        rules.loadFromUri(rule_path)

    transaction = Transaction(modsec, rules)

    # URI
    if verbose:
        print(method, full_url)
    transaction.processURI(full_url, method, "2.0")

    # Headers
    headers.append(f"Host: {parsed_url.netloc}")  # Avoid matching rule 920280
    for header in headers:
        name, value = header.split(":")
        transaction.addRequestHeader(name, value.strip())  # Avoid matching rule 920280
    transaction.processRequestHeaders()

    # Body
    if request_body:
        body = request_body.read().decode("utf-8")
        transaction.appendRequestBody(body)
        print(body)
    transaction.processRequestBody()

    activated_rules = []
    for rule in transaction.m_rulesMessages:
        if get_paranoia_level(rule) <= paranoia_level:
            activated_rules.append(rule.m_ruleId)

    return activated_rules

# test
# payload = "U0VMRUNUICogRlJPTSB1c2VycyBXSEVSRSBpZCA9IDE="
# print(base64.b64decode(payload).decode("utf-8"))
# matches = get_activated_rules([payload])
# print(matches)