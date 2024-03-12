#pragma once

#include <vector>
#include <fstream>

// 784
const int MNIST_size = 28 * 28;

template <typename T>
class mnist
{
    std::vector<char> trai_label;
    std::vector<std::vector<std::vector<T>>> trai_img;

    std::vector<char> test_label;
    std::vector<std::vector<std::vector<T>>> test_img;

public:
    mnist()
    {
        std::ifstream ifs;
        ifs.open("MNIST/train-images.idx3-ubyte", std::ios::binary);
        load_img(ifs, trai_img, 60000);
        ifs.close();

        ifs.open("MNIST/t10k-images.idx3-ubyte", std::ios::binary);
        load_img(ifs, test_img, 10000);
        ifs.close();

        ifs.open("MNIST/train-labels.idx1-ubyte", std::ios::binary);
        load_label(ifs, trai_label, 60000);
        ifs.close();

        ifs.open("MNIST/t10k-labels.idx1-ubyte", std::ios::binary);
        load_label(ifs, test_label, 10000);
        ifs.close();
    }

    std::vector<std::vector<T>> get_trai_img(const size_t id)
    {
        if (id >= trai_img.size())
        {
            return trai_img[0];
        }
        return trai_img[id];
    }

    std::vector<std::vector<T>> get_test_img(const size_t id)
    {
        if (id >= test_img.size())
        {
            return test_img[0];
        }
        return test_img[id];
    }

    std::vector<T> get_trai_label(const size_t id)
    {
        std::vector<T> temp(10, 0);
        if (id >= trai_label.size())
        {
            return temp;
        }
        temp[trai_label[id]] = 1;
        return temp;
    }

    std::vector<T> get_test_label(const size_t id)
    {
        std::vector<T> temp(10, 0);
        if (id >= test_label.size())
        {
            return temp;
        }
        temp[test_label[id]] = 1;
        return temp;
    }

private:
    void load_img(std::ifstream &ifs, std::vector<std::vector<std::vector<T>>> &arr, int size)
    {
        ifs.seekg(16, std::ios::beg);

        arr.resize(size, std::vector<std::vector<T>>(28, std::vector<T>(28)));

        std::vector<unsigned char> temp(MNIST_size);

        for (int i = 0; i < size; i++)
        {
            ifs.read(reinterpret_cast<char *>(&temp.front()), MNIST_size);
            for (int j = 0; j < 28; j++)
            {
                for (int k = 0; k < 28; k++)
                {
                    arr[i][j][k] = temp[j * 28 + k] / 255.0;
                }
            }
        }
    }

    void load_label(std::ifstream &ifs, std::vector<char> &arr, int size)
    {
        ifs.seekg(8, std::ios::beg);

        arr.resize(size);

        ifs.read(&arr.front(), size);
    }
};