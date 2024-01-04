# WAFcraft-docker
### Base Setup
```bash
# 1. Clone This Repo
git clone git@github.com:docluis/WAFcraft-docker.git
cd WAFcraft-docker
# 2. Clone Requirements, ModSecurity 3.0.10 release is already included
git clone git@github.com:pymodsecurity/pymodsecurity.git
git clone git@github.com:AvalZ/modsecurity-cli.git
git clone git@github.com:coreruleset/coreruleset.git
# 3. Checkout the pymodsecurity PR for ModSecurity 3.0.10 compatibility
cd pymodsecurity
gh pr checkout 21
# 4. Build the container and Attach!
docker-compose build; docker-compose run myapp bash
```
### Simple Test
```bash
# 5. Test
cd modsecurity-cli
python3 main.py "<script>alert(1)</script>" --verbose
```