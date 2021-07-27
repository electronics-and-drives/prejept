# Serving PRECEPT Models

Like [precppt](https://github.com/electronics-and-drives/precppt) but for java.

## Installation

Depends on [LibTorch](https://pytorch.org/cppdocs/installing.html)
and check `pom.xml` for further dependencies.

```bash
$ export LIBTORCH_HOME=/path/to/libtorch
$ export LD_LIBRARY_PATH="$LIBTORCH_HOME/lib"

$ mvn install
```

where `LIBTORCH_HOME` should be

```
/path/to/libtorch
├── bin
├── build-hash
├── build-version
├── include
├── lib
└── share
```
