#ifndef DDLA_TEST_MAX_ERROR_H
#define DDLA_TEST_MAX_ERROR_H

#include <algorithm>
#include <cmath>
#include <limits>

namespace ddla_test {

inline void update_max_error(double& error, double difference)
{
    error = std::isfinite(difference) ? std::max(error, difference)
                                      : std::numeric_limits<double>::infinity();
}

} // namespace ddla_test

#endif
