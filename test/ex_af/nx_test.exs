defmodule ExAF.NxTest do
  use ExAF.Case, async: true

  alias ExAF.Helpers

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

  @real_types @supported_types -- [{:c, 64}, {:c, 128}]

  @floats [{:f, 16}, {:f, 32}, {:f, 64}]

  @integers @real_types -- @floats

  @unary_ops [
    :exp,
    :log,
    :sin,
    :tan,
    :tanh,
    :atan,
    :asinh,
    :acosh,
    :erf,
    :sqrt,
    :abs,
    :floor,
    :real
  ]

  @binary_ops Helpers.binary_ops() -- [:left_shift, :right_shift]

  @rounding_unary_ops [
    :expm1,
    :log1p,
    :sigmoid,
    :cos,
    :sinh,
    :cosh,
    :erfc,
    :rsqrt,
    :cbrt,
    :round,
    :ceil
  ]

  # Backend management

  test "backend_copy/3" do
    backend =
      [1, 2, 3]
      |> Nx.tensor()
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Map.get(:data)
      |> Map.get(:__struct__)

    assert backend == Nx.BinaryBackend
  end

  describe "backend_deallocate/1" do
    test "handles regular deallocation" do
      response =
        [1, 2, 3]
        |> Nx.tensor()
        |> Nx.backend_deallocate()

      assert response == :ok
    end

    test "handles edge-case deallocation" do
      t = Nx.tensor([1, 2, 3])

      Nx.backend_deallocate(t)

      assert Nx.backend_deallocate(t) == :ok
    end
  end

  test "backend_transfer/2" do
    backend =
      [1, 2, 3]
      |> Nx.tensor()
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Map.get(:data)
      |> Map.get(:__struct__)

    assert backend == Nx.BinaryBackend
  end

  # Creation

  describe "tensor" do
    for type <- @supported_types do
      test "creation of #{Nx.Type.to_string(type)}" do
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

      test "constant creation of #{Nx.Type.to_string(type)}" do
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

  test "eye" do
    t1 = Nx.eye({2, 2})
    t2 = Nx.tensor([[1, 0], [0, 1]])

    assert_equal(t1, t2)
  end

  # Elementwise

  describe "manual rounding error tests" do
    test "asinh/1" do
      assert_all_close(Nx.tensor(3), Nx.asinh(Nx.tensor(10.017874)))
    end

    test "acosh/1" do
      assert_all_close(Nx.tensor(3), Nx.acosh(Nx.tensor(10.06766)))
    end

    test "atanh/1" do
      assert_all_close(Nx.tensor(3), Nx.atanh(Nx.tensor(0.9950547)))
    end
  end

  describe "rounding error tests" do
    for op <- @rounding_unary_ops, type <- @real_types do
      test "#{op}(#{Nx.Type.to_string(type)})" do
        test_round_unary_op(unquote(op), unquote(type))
      end
    end

    for type_left <- @real_types,
        type_right <- @real_types do
      test "divide(#{Nx.Type.to_string(type_left)}, #{Nx.Type.to_string(type_right)})" do
        test_round_binary_op(:divide, unquote(type_left), unquote(type_right))
      end
    end
  end

  describe "unary ops" do
    for op <- @unary_ops, type <- @real_types do
      test "#{op}(#{Nx.Type.to_string(type)})" do
        test_unary_op(unquote(op), unquote(type))
      end
    end
  end

  describe "binary ops" do
    for op <- @binary_ops -- [:divide],
        type_left <- @real_types,
        type_right <- @real_types do
      test "#{op}(#{Nx.Type.to_string(type_left)}, #{Nx.Type.to_string(type_right)})" do
        test_binary_op(unquote(op), unquote(type_left), unquote(type_right))
      end
    end

    for op <- [:right_shift, :left_shift],
        type_left <- @integers,
        type_right <- @integers do
      test "#{op}(#{Nx.Type.to_string(type_left)}, #{Nx.Type.to_string(type_right)})" do
        test_binary_op(unquote(op), unquote(type_left), unquote(type_right))
      end
    end
  end

  # Type

  describe "as_type" do
    for type1 <- @supported_types, type2 <- @supported_types do
      test "#{Nx.Type.to_string(type1)} from #{Nx.Type.to_string(type2)}" do
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

  # Helper functions

  defp apply_unary_op(op, data, type) do
    t = Nx.tensor(data, type: type)
    r = Kernel.apply(Nx, op, [t])

    binary_t = Nx.backend_transfer(t)

    binary_r = Kernel.apply(Nx, op, [binary_t])
    {r, binary_r}
  end

  defp test_round_unary_op(op, data \\ [[1, 2], [3, 4]], type) do
    {r, binary_r} = apply_unary_op(op, data, type)

    assert_all_close(r, binary_r)
  end

  defp test_unary_op(op, data \\ [[1, 2], [3, 4]], type) do
    {r, binary_r} = apply_unary_op(op, data, type)

    assert_equal(r, binary_r)
  end

  defp apply_binary_op(op, data_left, data_right, type_left, type_right) do
    left = Nx.tensor(data_left, type: type_left)
    right = Nx.tensor(data_right, type: type_right)

    r = Kernel.apply(Nx, op, [left, right])

    bleft = Nx.backend_transfer(left)
    bright = Nx.backend_transfer(right)
    binary_r = Kernel.apply(Nx, op, [bleft, bright])

    {r, binary_r}
  end

  defp test_binary_op(
         op,
         data_left \\ [[5, 6], [7, 8]],
         data_right \\ [[1, 2], [3, 4]],
         type_left,
         type_right
       ) do
    {r, binary_r} = apply_binary_op(op, data_left, data_right, type_left, type_right)

    assert_equal(r, binary_r)
  end

  defp test_round_binary_op(
         op,
         data_left \\ [[5, 6], [7, 8]],
         data_right \\ [[1, 2], [3, 4]],
         type_left,
         type_right
       ) do
    {r, binary_r} = apply_binary_op(op, data_left, data_right, type_left, type_right)

    assert_all_close(r, binary_r)
  end
end
