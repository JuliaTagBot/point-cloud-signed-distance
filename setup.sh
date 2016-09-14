DIRECTORY=$(cd `dirname "$0"` && pwd)
export JULIA_LOAD_PATH="$HOME/.julia"
export JULIA_PKGDIR="$DIRECTORY/packages"
echo "Julia package directory set to: $JULIA_PKGDIR with additional packages from $JULIA_LOAD_PATH"
