import re
import os
import glob

rules_path = "/app/ml-modsec/rules"


def get_rules_list():
    # read rules from each file in the rules directory
    for rule_path in sorted(glob.glob(f"{rules_path}/*.conf")):
        with open(rule_path, "r") as rule_file:
            rules = rule_file.read()
            matches = re.findall(r"id:(\d+),", rules)
    # return sorted list of unique rule IDs
    return sorted(set(matches))


def payload_to_array(payload, rule_ids):
    # TODO: optimize, main.py can handle multiple payloads at once
    # alternatively, create your own Transaction processor or
    # create PR to modsecurity-cli, maybe not necessary

    # run modsecurity-cli/main.py with the payload
    stream = os.popen(
        f'python ../modsecurity-cli/main.py --verbose --rules {rules_path} "{payload}"'
    )
    output = stream.read()

    # parse IDs from output (in format -  942432)
    matches = re.findall(r"[+/-]\s+(\d+)", output)
    rule_array = [1 if rule_id in set(matches) else 0 for rule_id in rule_ids]

    return rule_array


# testing
rule_ids = get_rules_list()
payload = "' or 1=1 -- -"

print(payload_to_array(payload, rule_ids))
