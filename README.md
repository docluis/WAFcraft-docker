# WAFcraft-docker
### Base Setup
```bash
# 1. Clone This Repo
git clone git@github.com:docluis/WAFcraft-docker.git
cd WAFcraft-docker
# 2. Clone required Repos
git clone git@github.com:pymodsecurity/pymodsecurity.git
git clone git@github.com:AvalZ/modsecurity-cli.git
git clone git@github.com:coreruleset/coreruleset.git
git clone git@github.com:AvalZ/WAF-A-MoLE.git
git clone git@github.com:zangobot/wafamole_dataset.git
git clone git@github.com:Morzeux/HttpParamsDataset.git
# 3. Download and extract Modsecurity 3.0.10 release
wget https://github.com/SpiderLabs/ModSecurity/releases/download/v3.0.10/modsecurity-v3.0.10.tar.gz
tar -xzf modsecurity-v3.0.10.tar.gz
# 4. Checkout the pymodsecurity PR for ModSecurity 3.0.10 compatibility
cd pymodsecurity
gh pr checkout 21
# 5. Build the container and Attach!
docker-compose down; docker-compose build; docker-compose up -d; sleep 1; docker-compose exec wafcraft bash
```
### How to run modsecurity-cli
> Test if the requirements have been installed correctly
```bash
cd modsecurity-cli
python main.py --verbose "' or 1=1 -- -"
python main.py --verbose --rules /app/wafcraft/rules "' or 1=1 -- -"
```

### How to run wafcraft
```bash
cd wafcraft # navigate to wafcraft directory (in Docker container)
# example creation of Target config 
python main.py --data --config Target --new
# example creation of Surrogate with 0% data overlap
python main.py --data --config Surrogate_Data_V1 --new
# example rerun of data creation (incase of crash or to add more samples)
python main.py --data --config Surrogate_Data_V1 --workspace <WORKSPACE DIR>
# test transferability
python main.py --transfer --config Target --target <TARGET WORKSPACE DIR> --surrogate <SURROGATE WORKSPACE DIR>
# add this to notify if something went wrong
|| curl -d "`hostname`: something went wrong :/" ntfy.sh/luis-info-buysvauy12iiq
```
> Access the **Jupyter Notebooks** via: http://127.0.0.1:8888/?token=aC9Zsec4kHLAcYndnYoUsaZbM52LrT