#include "NN/NN.hpp"
#include "MNIST/MNIST.hpp"
#include "NN/io.hpp"

void test(int batch_size, int loop, double Learning_Rate)
{
    NN nn;
    nn.add_Layer(flatten({1, 28, 28}));

    nn.add_Layer(Dense(128, He));
    nn.add_Layer(ReLU());
    nn.add_Layer(Dense(64, He));
    nn.add_Layer(ReLU());
    nn.add_Layer(Dense(10, He));

    nn.set_Loss(Softmax_with_Loss());

    mnist<double> m;
    Random<std::uniform_int_distribution<>> r(0, 60000 - 1);
    for (int i = 0; i < loop; i++)
    {
        double loss = 0, a = 0;
        for (int i = 0; i < batch_size; ++i)
        {
            int R = r();
            vvvd x(1, vvd(m.get_trai_img(R)));
            vd t = m.get_trai_label(R);
            vvvd y = nn.predict(x);
            if (std::distance(t.begin(), std::max_element(t.begin(), t.end())) == std::distance(y[0][0].begin(), std::max_element(y[0][0].begin(), y[0][0].end())))
            {
                a++;
            }
            loss += nn.gradient(x, t);
        }
        nn.update(Learning_Rate / batch_size);
        out("loss=", loss / batch_size);
        newline();
        out("正解:", a);
        newline();
    }

    double loss = 0, a = 0;
    for (int i = 0; i < 10000; ++i)
    {
        vvvd x(1, vvd(m.get_test_img(i)));
        vd t = m.get_test_label(i);
        vvvd y = nn.predict(x);
        if (std::distance(t.begin(), std::max_element(t.begin(), t.end())) == std::distance(y[0][0].begin(), std::max_element(y[0][0].begin(), y[0][0].end())))
        {
            a++;
        }
    }
    out("正解発表");
    newline();
    out("loss=", loss / 10000);
    newline();
    out("正解:", a);
    newline();
}

int main()
{
    test(100, 100, 0.01);
}