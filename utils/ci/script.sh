# This script takes care of testing your crate

set -ex

main() {
    # TODO: tweak what builds/tests we do where
    
    cross build --all --no-default-features --target $TARGET --release
    if [ ! -z $DISABLE_STD ]; then
        return
    fi
    
    cross build --target $TARGET
    if [ ! -z $NIGHTLY ]; then
        cross doc --no-deps --features nightly
    fi

    if [ ! -z $DISABLE_TESTS ]; then
        return
    fi

    cross test --tests --no-default-features --target $TARGET
    cross test --all --target $TARGET

    if [ ! -z $NIGHTLY ]; then
        cross test --all --features nightly --target $TARGET
        cross test --tests --no-default-features --features=alloc --target $TARGET
        cross test --all --benches --target $TARGET
    fi
}

# we don't run the "test phase" when doing deploys
if [ -z $TRAVIS_TAG ]; then
    main
fi