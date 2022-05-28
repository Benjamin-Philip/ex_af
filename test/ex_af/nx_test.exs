defmodule ExAF.NxTest do
  use ExAF.Case, async: true

  @supported_types [
    {:u, 8},
    {:u, 16},
    {:u, 32},
    {:u, 64},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:f, 16},
    {:f, 32},
    {:f, 64},
    {:c, 64},
    {:c, 128}
  ]

  # Creation

  describe "tensor" do
    for type <- @supported_types do
      test "creation of #{inspect(type)}" do
        list = [1, 2, 3]

        exaf_binary =
          list
          |> Nx.tensor(type: unquote(type))
          |> Nx.to_binary()

        binary_backend_binary =
          list
          |> Nx.tensor(type: unquote(type), backend: Nx.BinaryBackend)
          |> Nx.to_binary()

        assert exaf_binary == binary_backend_binary
      end

      test "constant creation of #{inspect(type)}" do
        scalar = 1

        exaf_binary =
          scalar
          |> Nx.tensor(type: unquote(type))
          |> Nx.to_binary()

        binary_backend_binary =
          scalar
          |> Nx.tensor(type: unquote(type), backend: Nx.BinaryBackend)
          |> Nx.to_binary()

        assert exaf_binary == binary_backend_binary
      end
    end
  end

  describe "iota" do
    test "with a constant" do
      t1 = Nx.iota({})
      t2 = Nx.tensor(0)

      assert_equal(t1, t2)
    end

    test "without an axis" do
      t1 = Nx.iota({2, 2})
      t2 = Nx.tensor([[0, 1], [2, 3]])

      assert_equal(t1, t2)
    end

    test "with an axis one less than rank" do
      t1 = Nx.iota({2, 2}, axis: 1)
      t2 = Nx.tensor([[0, 1], [0, 1]])

      assert_equal(t1, t2)
    end

    test "with an axis" do
      t1 = Nx.iota({2, 2}, axis: 0)
      t2 = Nx.tensor([[0, 0], [1, 1]])

      assert_equal(t1, t2)
    end
  end

  # Type

  describe "as_type" do
    for type1 <- @supported_types, type2 <- @supported_types do
      test "#{inspect(type1)} from #{inspect(type2)}" do
        list = [1, 2, 3]

        t1 =
          list
          |> Nx.tensor(type: unquote(type1))
          |> Nx.as_type(unquote(type2))

        t2 = Nx.tensor(list, type: unquote(type2))

        assert_equal(t1, t2)
      end
    end
  end
end
