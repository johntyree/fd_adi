#ifndef TILED_RANGE_H
#define TILED_RANGE_H

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

/* create tiled_range with elements tiled x3
 *
 * typedef thrust::device_vector<int>::iterator Iterator;
 * tiled_range<Iterator> thrice(data.begin(), data.end(), 3);
 */

template <typename Iterator>
class tiled_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;
    struct tile_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type tiles;
        Iterator first;
        Iterator last;


        tile_functor(Iterator first, Iterator last, difference_type tiles)
            : first(first), last(last), tiles(tiles) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return i % (last - first);
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the tiled_range iterator
    typedef PermutationIterator iterator;

    // construct tiled_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type tiles)
        : first(first), last(last), tiles(tiles) {}

    iterator begin(void) const
    {
        return PermutationIterator(first,
                TransformIterator(CountingIterator(0),
                    tile_functor(first, last, tiles)));
    }

    iterator end(void) const
    {
        return begin() + tiles * (last - first);
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type tiles;
};

#endif /* end of include guard */
