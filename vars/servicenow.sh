# for workarena:
export SNOW_INSTANCE_URL="https://dev275972.service-now.com"
export SNOW_INSTANCE_UNAME="admin"
export SNOW_INSTANCE_PWD="<password>"

# if PWD is not set, raise an error
if [ -z "$SNOW_INSTANCE_PWD" ]; then
  echo "Error: SNOW_INSTANCE_PWD is not set. Please set it in the vars/servicenow.sh file."
  exit 1
fi