import base64
import re
import os
import sqlparse
import glob
import shlex
import subprocess

rules_path = "/app/ml-modsec/rules"
wafamole_dataset_path = "/app/wafamole_dataset"


def get_rules_list():
    # TODO: maybe do this statically, once ?

    # read rules from each file in the rules directory
    for rule_path in sorted(glob.glob(f"{rules_path}/*.conf")):
        with open(rule_path, "r") as rule_file:
            rules = rule_file.read()
            matches = re.findall(r"id:(\d+),", rules)
    # return sorted list of unique rule IDs
    return sorted(set(matches))


def payload_to_vec(payload, rule_ids):
    # TODO: optimize, main.py can handle multiple payloads at once
    # alternatively, create your own Transaction processor or
    # create PR to modsecurity-cli, maybe not necessary

    # escape payload
    payload = shlex.quote(payload)
    # run modsecurity-cli/main.py with the payload
    full_command = (
        f"python ../modsecurity-cli/main.py --verbose --rules {rules_path} {payload}"
    )

    process = subprocess.Popen(
        full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = process.communicate()
    # parse IDs from output (in format -  942432)
    matches = re.findall(r"\s+(\d+)", output.decode("utf-8"))
    rule_array = [1 if rule_id in set(matches) else 0 for rule_id in rule_ids]

    return rule_array


def generate_dataset(size=1000):
    rule_ids = get_rules_list()

    if not os.path.exists("data/attacks.csv"):
        os.makedirs("data")
        full_command = f"cat {wafamole_dataset_path}/attacks.sql.* > data/attacks.csv"
        process = subprocess.Popen(
            full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()
        print("attacks.csv created")

    if not os.path.exists("data/sanes.csv"):
        full_command = f"cat {wafamole_dataset_path}/sane.sql.* > data/sanes.csv"
        process = subprocess.Popen(
            full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()
        print("sanes.csv created")

    with open("data/attacks.csv", "r") as attacks_file:
        lim = int(size / 2) + 1
        # only sample size lines
        attacks_b = attacks_file.readlines()[:lim] # TODO: sample random ones maybe?
        attacks = sqlparse.split("".join(attacks_b))
        for attack in attacks:
            vec = payload_to_vec(attack, rule_ids)
            base64_attack = base64.b64encode(attack.encode("utf-8")).decode("utf-8")
            # write to file
            with open("train_dataset.csv", "a") as train_dataset_file:
                train_dataset_file.write(f"{base64_attack},{vec},1\n")
    
    with open("data/sanes.csv", "r") as sanes_file:
        lim = int(size / 2) + 1
        # only sample size lines
        sanes_b = sanes_file.readlines()[:lim] # TODO: sample random ones maybe?
        sanes = sqlparse.split("".join(sanes_b))
        for sane in sanes:
            vec = payload_to_vec(sane, rule_ids)
            base64_sane = base64.b64encode(sane.encode("utf-8")).decode("utf-8")
            # write to file
            with open("train_dataset.csv", "a") as train_dataset_file:
                train_dataset_file.write(f"{base64_sane},{vec},0\n")


# testing
# rule_ids = get_rules_list()
# payload = "' or 1=1 -- -"
# payload = shlex.quote("SELECT `col2` FROM `tab` WHERE `col1` LIKE '%'f'%';")

# print(payload_to_vec(payload, rule_ids))

generate_dataset(10)
