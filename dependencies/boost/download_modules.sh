#!/bash/bin

git submodule update --depth 1 -q --init tools/boostdep

python_cmd=$1
echo "python executable has been set to ${python_cmd}"

boost_libs="iostreams json math headers serialization system thread"

for lib in $boost_libs; do
  echo "Setting up boost component ${lib}..."
  git submodule update --depth 1 -q --init libs/${lib}
  ${python_cmd} tools/boostdep/depinst/depinst.py -X test -g "--depth 1" ${lib}
done

