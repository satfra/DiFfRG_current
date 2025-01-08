mkdir -p $SCRIPT_PATH/logs
mkdir -p $BUILD_PATH

failed_first=0
mkdir -p $INSTALL_PATH &>/dev/null && touch $INSTALL_PATH/_permission_test &>/dev/null || {
  failed_first=1
}

SuperUser=""
if ((failed_first == 0)); then
  echo "Elevated permissions required to write into $INSTALL_PATH."
  SuperUser="sudo -E"

  $SuperUser mkdir -p $INSTALL_PATH
else
  rm $INSTALL_PATH/_permission_test
fi
