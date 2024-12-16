THREADS='1'
CLEANUP_FLAG=false
while getopts "cj:" flag; do
  case "${flag}" in
  j) THREADS=${OPTARG}
      ;;
  c) CLEANUP_FLAG=true
      ;;
  esac
done
