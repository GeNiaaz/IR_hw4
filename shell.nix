{ pkgs ? import <nixpkgs> {} }:

with pkgs;
let
  my-python-packages = python-packages: with python-packages; [
    nltk
    numpy
    getopt
    scikitlearn
    joblib
    PyStemmer
    linecache2
  ];
  python-with-my-packages = python3.withPackages my-python-packages;
in

stdenv.mkDerivation {
  name = "cs3245";
  buildInputs = [ python-with-my-packages ];
}
