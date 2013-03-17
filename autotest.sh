#!/bin/bash

onmodify () {
    TARGET=${1:-.};
    [[ $ALL ]] && rm -f .noseids
    while inotifywait -qq -r -e close_write,moved_to,move_self $TARGET; do
        sleep 0.5;
        build && run;
    done
}

while getopts ":acfb" opt; do
    case $opt in
        a)
            ALL="remove_ids"
            ;;
        c)
            CCC="force_rebuild;"
            ;;
        f)
            failed="--failed"
            ;;
        b)
            $CCC
            python setup.py build_ext --inplace 2>&1
            exit
            ;;
    esac
done
shift $(($OPTIND-1))

build () {
    [[ $CCC ]] && (for i in FiniteDifference/*.pyx; do rm ${i%%.pyx}.cpp; done)
    set -o pipefail
    python setup.py build_ext --inplace 2>&1 | grep -Ei --color -e '' -e error
}

run () {
    nosetests $failed --rednose --verbosity=3 --with-id $@ || echo -ne '\a'
}

onmodify .
