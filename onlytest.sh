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
        sleep 0.2;
        if [ "$CMD" ]; then
            bash -c "$CMD";
        else
            build && run;
        fi;
        echo;
    done )
}

# onmodify . "touch FiniteDifference/_GPU_Code.cu; python setup.py nosetests --failed --rednose --verbosity=3 --with-id $@ || echo -ne '\a'"
onmodify . "python setup.py build_ext --inplace && nosetests --rednose --verbosity=3 --with-id $@ || echo -ne '\a'"