#!/bin/bash

onmodify () {
    TARGET=${1:-.};
    shift;
    CMD="$@";
    echo "$TARGET" "$CMD";
    ( if [ -f onmodify.sh ]; then
        . onmodify.sh;
    fi;
    while inotifywait -qq -r -e close_write,moved_to,move_self $TARGET; do
        sleep 0.5;
        if [ "$CMD" ]; then
            bash -c "$CMD";
        else
            build && run;
        fi;
        echo;
    done )
}

ALL=""
CCC=""
while getopts ":acfb" opt; do
    case $opt in
        a)
            ALL="rm .noseids;"
            ;;
        c)
            CCC="touch $(ls FiniteDifference/*Code.cu);"
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

CMD="$ALL $CCC (set -o pipefail; python setup.py build_ext --inplace 2>&1 | grep -Ei --color -e '' -e error) \
    && nosetests $failed --rednose --verbosity=3 --with-id $@ \
    || echo -ne '\a'
"

onmodify . $CMD
