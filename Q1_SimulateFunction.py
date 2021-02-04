{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled6.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPEhOWxfsmqIIVYChP6NSLz",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ktripat/demo-repo/blob/main/Q1_SimulateFunction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2UPjiyWQbQaY"
      },
      "source": [
        "import numpy as np\r\n",
        "import pandas as pd\r\n",
        "import tensorflow as tf \r\n",
        "from matplotlib import pyplot as plt\r\n",
        "import seaborn as sns\r\n"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "N_4KMc_Jqcrz"
      },
      "source": [
        "The nonlinear function considered here : ***f(x)= -x + 5 sin(x)***.\r\n",
        "\r\n",
        "*   Input range of x: (0,10pi) for the training datasets \r\n",
        "\r\n",
        "Let's show the function using the scatter plot."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "jZTwIzAwbw6T",
        "outputId": "c44d552f-dd31-45c4-890c-e39c7eba2019"
      },
      "source": [
        "x=np.linspace(0,10*np.pi,500)\r\n",
        "y=np.exp(-x)\r\n",
        "plt.scatter(x,y)"
      ],
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7fc6b2c181d0>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 47
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "PWKWR-o4McGX",
        "outputId": "621fb0e4-c359-4632-dfed-b7662c2b4609"
      },
      "source": [
        "model_shallow=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(1,)),\r\n",
        "                                  tf.keras.layers.Dense(128,activation=tf.nn.relu),\r\n",
        "                                  tf.keras.layers.Dropout(0.2),\r\n",
        "                                  tf.keras.layers.Dense(1)])\r\n",
        "model_shallow.compile(optimizer='adam',loss='mse',metrics=['accuracy'])\r\n",
        "hh=model_shallow.fit(x,y,epochs=100)\r\n",
        "model_shallow.summary()\r\n",
        "\r\n",
        "yp=model_shallow.predict(x)\r\n",
        "\r\n",
        "plt.scatter(x,y)\r\n",
        "plt.scatter(x,yp)"
      ],
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 1.3545 - accuracy: 0.0000e+00\n",
            "Epoch 2/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.9675 - accuracy: 0.0000e+00\n",
            "Epoch 3/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.7553 - accuracy: 0.0000e+00\n",
            "Epoch 4/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.6467 - accuracy: 0.0000e+00\n",
            "Epoch 5/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.6530 - accuracy: 0.0000e+00\n",
            "Epoch 6/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.6191 - accuracy: 0.0000e+00\n",
            "Epoch 7/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.5277 - accuracy: 0.0000e+00\n",
            "Epoch 8/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.4816 - accuracy: 0.0000e+00\n",
            "Epoch 9/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.5449 - accuracy: 0.0000e+00\n",
            "Epoch 10/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.3540 - accuracy: 0.0000e+00\n",
            "Epoch 11/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.3612 - accuracy: 0.0000e+00\n",
            "Epoch 12/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.3268 - accuracy: 0.0000e+00\n",
            "Epoch 13/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.3469 - accuracy: 0.0000e+00\n",
            "Epoch 14/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.3021 - accuracy: 0.0000e+00\n",
            "Epoch 15/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.1867 - accuracy: 0.0000e+00\n",
            "Epoch 16/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.2228 - accuracy: 0.0000e+00\n",
            "Epoch 17/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.2440 - accuracy: 0.0000e+00\n",
            "Epoch 18/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.2018 - accuracy: 0.0000e+00\n",
            "Epoch 19/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.1744 - accuracy: 0.0000e+00\n",
            "Epoch 20/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.1565 - accuracy: 0.0000e+00\n",
            "Epoch 21/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.1466 - accuracy: 0.0000e+00\n",
            "Epoch 22/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.1519 - accuracy: 0.0000e+00\n",
            "Epoch 23/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.1322 - accuracy: 0.0000e+00\n",
            "Epoch 24/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0992 - accuracy: 0.0000e+00\n",
            "Epoch 25/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.1197 - accuracy: 0.0000e+00\n",
            "Epoch 26/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0963 - accuracy: 0.0000e+00\n",
            "Epoch 27/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0871 - accuracy: 0.0000e+00\n",
            "Epoch 28/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.1013 - accuracy: 0.0000e+00\n",
            "Epoch 29/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0696 - accuracy: 0.0000e+00\n",
            "Epoch 30/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0669 - accuracy: 0.0000e+00\n",
            "Epoch 31/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0671 - accuracy: 0.0000e+00\n",
            "Epoch 32/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0659 - accuracy: 0.0000e+00\n",
            "Epoch 33/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0708 - accuracy: 0.0000e+00\n",
            "Epoch 34/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0589 - accuracy: 0.0000e+00\n",
            "Epoch 35/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0515 - accuracy: 0.0000e+00\n",
            "Epoch 36/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0505 - accuracy: 0.0000e+00\n",
            "Epoch 37/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0459 - accuracy: 0.0000e+00\n",
            "Epoch 38/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0446 - accuracy: 0.0000e+00\n",
            "Epoch 39/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0421 - accuracy: 0.0000e+00\n",
            "Epoch 40/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0428 - accuracy: 0.0000e+00\n",
            "Epoch 41/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0405 - accuracy: 0.0000e+00\n",
            "Epoch 42/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0421 - accuracy: 0.0000e+00\n",
            "Epoch 43/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0358 - accuracy: 0.0000e+00\n",
            "Epoch 44/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0364 - accuracy: 0.0000e+00\n",
            "Epoch 45/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0278 - accuracy: 0.0000e+00\n",
            "Epoch 46/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0275 - accuracy: 0.0000e+00\n",
            "Epoch 47/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0324 - accuracy: 0.0000e+00\n",
            "Epoch 48/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0342 - accuracy: 0.0000e+00\n",
            "Epoch 49/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0261 - accuracy: 0.0000e+00\n",
            "Epoch 50/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0306 - accuracy: 0.0000e+00\n",
            "Epoch 51/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0275 - accuracy: 0.0000e+00\n",
            "Epoch 52/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0240 - accuracy: 0.0000e+00\n",
            "Epoch 53/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0248 - accuracy: 0.0000e+00\n",
            "Epoch 54/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0263 - accuracy: 0.0000e+00\n",
            "Epoch 55/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0219 - accuracy: 0.0000e+00\n",
            "Epoch 56/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0201 - accuracy: 0.0000e+00\n",
            "Epoch 57/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0191 - accuracy: 0.0000e+00\n",
            "Epoch 58/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0259 - accuracy: 0.0000e+00\n",
            "Epoch 59/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0244 - accuracy: 0.0000e+00\n",
            "Epoch 60/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0194 - accuracy: 0.0000e+00\n",
            "Epoch 61/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0207 - accuracy: 0.0000e+00\n",
            "Epoch 62/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0207 - accuracy: 0.0000e+00\n",
            "Epoch 63/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0206 - accuracy: 0.0000e+00\n",
            "Epoch 64/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0139 - accuracy: 0.0000e+00\n",
            "Epoch 65/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0197 - accuracy: 0.0000e+00\n",
            "Epoch 66/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0205 - accuracy: 0.0000e+00\n",
            "Epoch 67/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0188 - accuracy: 0.0000e+00\n",
            "Epoch 68/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0136 - accuracy: 0.0000e+00\n",
            "Epoch 69/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0142 - accuracy: 0.0000e+00\n",
            "Epoch 70/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0152 - accuracy: 0.0000e+00\n",
            "Epoch 71/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0128 - accuracy: 0.0000e+00\n",
            "Epoch 72/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0167 - accuracy: 0.0000e+00\n",
            "Epoch 73/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0177 - accuracy: 0.0000e+00\n",
            "Epoch 74/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0162 - accuracy: 0.0000e+00\n",
            "Epoch 75/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0145 - accuracy: 0.0000e+00\n",
            "Epoch 76/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0152 - accuracy: 0.0000e+00\n",
            "Epoch 77/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0129 - accuracy: 0.0000e+00\n",
            "Epoch 78/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0169 - accuracy: 0.0000e+00\n",
            "Epoch 79/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0163 - accuracy: 0.0000e+00\n",
            "Epoch 80/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0093 - accuracy: 0.0000e+00\n",
            "Epoch 81/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0117 - accuracy: 0.0000e+00\n",
            "Epoch 82/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0131 - accuracy: 0.0000e+00\n",
            "Epoch 83/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0118 - accuracy: 0.0000e+00\n",
            "Epoch 84/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0109 - accuracy: 0.0000e+00\n",
            "Epoch 85/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0131 - accuracy: 0.0000e+00\n",
            "Epoch 86/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0142 - accuracy: 0.0000e+00\n",
            "Epoch 87/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0075 - accuracy: 0.0000e+00\n",
            "Epoch 88/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0121 - accuracy: 0.0000e+00\n",
            "Epoch 89/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0098 - accuracy: 0.0000e+00\n",
            "Epoch 90/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0140 - accuracy: 0.0000e+00\n",
            "Epoch 91/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0100 - accuracy: 0.0000e+00\n",
            "Epoch 92/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0109 - accuracy: 0.0000e+00\n",
            "Epoch 93/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0087 - accuracy: 0.0000e+00\n",
            "Epoch 94/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0106 - accuracy: 0.0000e+00\n",
            "Epoch 95/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0144 - accuracy: 0.0000e+00\n",
            "Epoch 96/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0130 - accuracy: 0.0000e+00\n",
            "Epoch 97/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0116 - accuracy: 0.0000e+00\n",
            "Epoch 98/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0179 - accuracy: 0.0000e+00\n",
            "Epoch 99/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0108 - accuracy: 0.0000e+00\n",
            "Epoch 100/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0130 - accuracy: 0.0000e+00\n",
            "Model: \"sequential_19\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "flatten_5 (Flatten)          (None, 1)                 0         \n",
            "_________________________________________________________________\n",
            "dense_66 (Dense)             (None, 128)               256       \n",
            "_________________________________________________________________\n",
            "dropout_16 (Dropout)         (None, 128)               0         \n",
            "_________________________________________________________________\n",
            "dense_67 (Dense)             (None, 1)                 129       \n",
            "=================================================================\n",
            "Total params: 385\n",
            "Trainable params: 385\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7fc6ae4a21d0>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 48
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nO6Ka7s1r7_K"
      },
      "source": [
        "**Model #1 Shallow neural network (using one hidden layer)**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 285
        },
        "id": "0vpTXjWALycg",
        "outputId": "572eeb78-3104-4735-88bd-4c3d596918a1"
      },
      "source": [
        "## new testing data\r\n",
        "x_test=np.arange(2.2,5.6,0.1)\r\n",
        "y_test=model_shallow.predict(x_test)\r\n",
        "\r\n",
        "plt.scatter(x_test,np.exp(-x_test))\r\n",
        "plt.scatter(x_test,y_test)"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7fc6ac1c7080>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 49
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DQD_6ZFhcba6"
      },
      "source": [
        "**Model #2 Deep neural network (using 3 hidden layers)**\r\n",
        "\r\n",
        "\r\n",
        "\r\n",
        "\r\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nm_gn-2XcOhq",
        "outputId": "d4daa005-d517-49e5-883e-beb07c92b4d2"
      },
      "source": [
        "model_deep=tf.keras.models.Sequential([#tf.keras.layers.Flatten(x),\r\n",
        "                                  tf.keras.layers.Dense(64,activation=tf.nn.relu),\r\n",
        "                                  tf.keras.layers.Dropout(0.2),\r\n",
        "                                  tf.keras.layers.Dense(128,activation=tf.nn.relu),\r\n",
        "                                  tf.keras.layers.Dense(64,activation=tf.nn.relu),\r\n",
        "                                  tf.keras.layers.Dense(1)])\r\n",
        "#opt = tf.keras.optimizers.Adam(learning_rate=0.02)\r\n",
        "model_deep.compile(optimizer='adam',loss='mse',metrics=['accuracy'])\r\n",
        "hh=model_deep.fit(x,y,epochs=100)\r\n",
        "model_deep.summary()\r\n",
        "\r\n",
        "\r\n",
        "\r\n",
        "\r\n",
        "\r\n",
        "\r\n"
      ],
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.7814 - accuracy: 0.0000e+00\n",
            "Epoch 2/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.1136 - accuracy: 0.0000e+00\n",
            "Epoch 3/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0583 - accuracy: 0.0000e+00\n",
            "Epoch 4/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0604 - accuracy: 0.0000e+00\n",
            "Epoch 5/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0518 - accuracy: 0.0000e+00\n",
            "Epoch 6/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0497 - accuracy: 0.0000e+00\n",
            "Epoch 7/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0393 - accuracy: 0.0000e+00\n",
            "Epoch 8/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0418 - accuracy: 0.0000e+00\n",
            "Epoch 9/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0302 - accuracy: 0.0000e+00\n",
            "Epoch 10/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0233 - accuracy: 0.0000e+00\n",
            "Epoch 11/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0181 - accuracy: 0.0000e+00\n",
            "Epoch 12/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0174 - accuracy: 0.0000e+00\n",
            "Epoch 13/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0153 - accuracy: 0.0016\n",
            "Epoch 14/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0126 - accuracy: 3.5784e-04\n",
            "Epoch 15/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0094 - accuracy: 0.0011\n",
            "Epoch 16/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0116 - accuracy: 3.5784e-04\n",
            "Epoch 17/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0093 - accuracy: 0.0018\n",
            "Epoch 18/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0110 - accuracy: 0.0025\n",
            "Epoch 19/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0087 - accuracy: 0.0018\n",
            "Epoch 20/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0077 - accuracy: 0.0063\n",
            "Epoch 21/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0067 - accuracy: 0.0025\n",
            "Epoch 22/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0081 - accuracy: 2.3529e-04\n",
            "Epoch 23/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0077 - accuracy: 0.0025\n",
            "Epoch 24/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0046 - accuracy: 0.0063\n",
            "Epoch 25/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0070 - accuracy: 6.3055e-04\n",
            "Epoch 26/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0050 - accuracy: 0.0063\n",
            "Epoch 27/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0038 - accuracy: 0.0036\n",
            "Epoch 28/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0050 - accuracy: 7.8373e-04\n",
            "Epoch 29/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0057 - accuracy: 3.5784e-04\n",
            "Epoch 30/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0059 - accuracy: 0.0016\n",
            "Epoch 31/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0041 - accuracy: 0.0030\n",
            "Epoch 32/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0050 - accuracy: 0.0030\n",
            "Epoch 33/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0038 - accuracy: 0.0045\n",
            "Epoch 34/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0037 - accuracy: 0.0063\n",
            "Epoch 35/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0033 - accuracy: 0.0045\n",
            "Epoch 36/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0049 - accuracy: 0.0013\n",
            "Epoch 37/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0032 - accuracy: 0.0045\n",
            "Epoch 38/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0030 - accuracy: 7.8373e-04\n",
            "Epoch 39/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0023 - accuracy: 0.0063\n",
            "Epoch 40/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0020 - accuracy: 7.8373e-04\n",
            "Epoch 41/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0039 - accuracy: 6.3055e-04\n",
            "Epoch 42/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0028 - accuracy: 4.8915e-04\n",
            "Epoch 43/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0033 - accuracy: 0.0063\n",
            "Epoch 44/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0023 - accuracy: 4.8915e-04\n",
            "Epoch 45/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0032 - accuracy: 2.3529e-04\n",
            "Epoch 46/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0024 - accuracy: 0.0018\n",
            "Epoch 47/100\n",
            "16/16 [==============================] - 0s 1ms/step - loss: 0.0025 - accuracy: 6.3055e-04\n",
            "Epoch 48/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0020 - accuracy: 2.3529e-04\n",
            "Epoch 49/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0023 - accuracy: 0.0030\n",
            "Epoch 50/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0017 - accuracy: 0.0013    \n",
            "Epoch 51/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0014 - accuracy: 4.8915e-04\n",
            "Epoch 52/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0014 - accuracy: 7.8373e-04\n",
            "Epoch 53/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0015 - accuracy: 0.0025\n",
            "Epoch 54/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0021 - accuracy: 0.0016\n",
            "Epoch 55/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0019 - accuracy: 0.0021\n",
            "Epoch 56/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0014 - accuracy: 2.3529e-04\n",
            "Epoch 57/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0013 - accuracy: 9.5085e-04\n",
            "Epoch 58/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0017 - accuracy: 0.0013\n",
            "Epoch 59/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0013 - accuracy: 0.0030    \n",
            "Epoch 60/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0018 - accuracy: 0.0063\n",
            "Epoch 61/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0017 - accuracy: 0.0036\n",
            "Epoch 62/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0013 - accuracy: 9.5085e-04\n",
            "Epoch 63/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0012 - accuracy: 0.0018\n",
            "Epoch 64/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0017 - accuracy: 4.8915e-04\n",
            "Epoch 65/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0011 - accuracy: 0.0045\n",
            "Epoch 66/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0010 - accuracy: 9.5085e-04\n",
            "Epoch 67/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0012 - accuracy: 4.8915e-04\n",
            "Epoch 68/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0017 - accuracy: 0.0063\n",
            "Epoch 69/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0012 - accuracy: 0.0018\n",
            "Epoch 70/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 9.9814e-04 - accuracy: 3.5784e-04\n",
            "Epoch 71/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0022 - accuracy: 4.8915e-04\n",
            "Epoch 72/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0017 - accuracy: 0.0025    \n",
            "Epoch 73/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 8.5823e-04 - accuracy: 6.3055e-04\n",
            "Epoch 74/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 9.9384e-04 - accuracy: 0.0045\n",
            "Epoch 75/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 8.4235e-04 - accuracy: 0.0036\n",
            "Epoch 76/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 7.7098e-04 - accuracy: 3.5784e-04\n",
            "Epoch 77/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0011 - accuracy: 7.8373e-04\n",
            "Epoch 78/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 9.6502e-04 - accuracy: 0.0045\n",
            "Epoch 79/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0011 - accuracy: 0.0063\n",
            "Epoch 80/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0011 - accuracy: 3.5784e-04\n",
            "Epoch 81/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0012 - accuracy: 0.0011\n",
            "Epoch 82/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0010 - accuracy: 4.8915e-04\n",
            "Epoch 83/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 9.5689e-04 - accuracy: 0.0021\n",
            "Epoch 84/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0012 - accuracy: 7.8373e-04\n",
            "Epoch 85/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 5.8008e-04 - accuracy: 0.0016\n",
            "Epoch 86/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0011 - accuracy: 0.0036\n",
            "Epoch 87/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 7.1627e-04 - accuracy: 9.5085e-04\n",
            "Epoch 88/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 6.5598e-04 - accuracy: 6.3055e-04\n",
            "Epoch 89/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 9.4073e-04 - accuracy: 4.8915e-04\n",
            "Epoch 90/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 8.3262e-04 - accuracy: 0.0045\n",
            "Epoch 91/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0010 - accuracy: 0.0018\n",
            "Epoch 92/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0015 - accuracy: 0.0063\n",
            "Epoch 93/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 0.0014 - accuracy: 0.0045\n",
            "Epoch 94/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 5.3047e-04 - accuracy: 0.0021\n",
            "Epoch 95/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 5.7568e-04 - accuracy: 2.3529e-04\n",
            "Epoch 96/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 4.4056e-04 - accuracy: 3.5784e-04\n",
            "Epoch 97/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 6.1751e-04 - accuracy: 0.0013\n",
            "Epoch 98/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 7.6219e-04 - accuracy: 0.0016\n",
            "Epoch 99/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 5.9765e-04 - accuracy: 0.0016\n",
            "Epoch 100/100\n",
            "16/16 [==============================] - 0s 2ms/step - loss: 5.4110e-04 - accuracy: 0.0030\n",
            "Model: \"sequential_20\"\n",
            "_________________________________________________________________\n",
            "Layer (type)                 Output Shape              Param #   \n",
            "=================================================================\n",
            "dense_68 (Dense)             (None, 64)                128       \n",
            "_________________________________________________________________\n",
            "dropout_17 (Dropout)         (None, 64)                0         \n",
            "_________________________________________________________________\n",
            "dense_69 (Dense)             (None, 128)               8320      \n",
            "_________________________________________________________________\n",
            "dense_70 (Dense)             (None, 64)                8256      \n",
            "_________________________________________________________________\n",
            "dense_71 (Dense)             (None, 1)                 65        \n",
            "=================================================================\n",
            "Total params: 16,769\n",
            "Trainable params: 16,769\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tqLiDCm5gfrW",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "f1d8189d-0510-48a2-cd6d-5878fbed0878"
      },
      "source": [
        "plt.plot(hh.history['loss'])\r\n",
        "plt.title('Model loss')\r\n",
        "plt.ylabel('Loss')\r\n",
        "plt.xlabel('Epoch')\r\n",
        "plt.show()\r\n"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Jk1ZRaf1cZfS"
      },
      "source": [
        "\r\n"
      ],
      "execution_count": 27,
      "outputs": []
    }
  ]
}