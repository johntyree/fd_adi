#!/bin/bash

onmodify () {
    TARGET=${1:-.};
    shift
    remove_ids
    build && run "${@}";
    while inotifywait -qq -r -e close_write,moved_to,move_self $TARGET; do
        sleep 0.5;
        build && run "${@}";
    done
}

remove_ids () {
    [[ $ALL ]] && rm -f .noseids
}

force_rebuild () {
    (for i in FiniteDifference/*.pyx; do rm ${i%%.pyx}.cpp; done)
}

build () {
    [[ $CCC ]] && force_rebuild
    set -o pipefail
    python setup.py build_ext --inplace 2>&1 | grep -Ei --color -e '' -e error
}

run () {
    args=($failed --rednose --verbosity=3 --with-id "$@")
    if [[ $USE_GDB ]]; then
        echo "gdb --args python $(which nosetests) ${args[@]} <<< run"
        gdb --args python $(which nosetests) ${args[@]} <<< run
    else
        echo "nosetests ${args[@]}"
        nosetests ${args[@]} || echo -ne '\a'
    fi
}

ARGS=()
while getopts ":acfgb" opt; do
    case $opt in
        a)
            ALL="remove_ids"
            ;;
        c)
            CCC="force_rebuild"
            ;;
        f)
            failed="--failed"
            ;;
        b)
            build
            exit
            ;;
        g)
            USE_GDB=1
            ;;
        \?)
            ARGS+=(-$OPTARG)
            ;;
    esac
done
shift $(($OPTIND-1))

onmodify . "$@"
