#pragma once

#include "../Layer.hpp"

class flatten : public Layer
{
    size_t X, Y, Z;

public:
    flatten(const tsss &input_size = {0, 0, 0})
    {
        if (std::get<0>(input_size) != 0 and std::get<1>(input_size) != 0 and std::get<2>(input_size) != 0)
        {
            set(input_size);
        }
    }

    void set(const tsss &input_size) override
    {
        X = std::get<0>(input_size);
        Y = std::get<1>(input_size);
        Z = std::get<2>(input_size);
    }
    tsss get() override { return {1, 1, X * Y * Z}; }

    vvvd forward(const vvvd &x) override
    {
        vvvd y(1, vvd(1, vd(X * Y * Z)));

        for (size_t i = 0; i < X; ++i)
        {
            for (size_t j = 0; j < Y; ++j)
            {
                for (size_t k = 0; k < Z; ++k)
                {
                    y[0][0][i * Y * Z + j * Z + k] = x[i][j][k];
                }
            }
        }
        return y;
    }

    vvvd backward(const vvvd &dy) override
    {
        vvvd dx(X, vvd(Y, vd(Z)));

        for (size_t i = 0; i < X; ++i)
        {
            for (size_t j = 0; j < Y; ++j)
            {
                for (size_t k = 0; k < Z; ++k)
                {
                    dx[i][j][k] = dy[0][0][i * Y * Z + j * Z + k];
                }
            }
        }
        return dx;
    }
};