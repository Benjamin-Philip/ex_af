defmodule ExAF.NxTest do
  use ExUnit.Case

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

  describe "tensor" do
    test "creation" do
      for type <- @supported_types do
        list = [1, 2, 3]

        exaf_binary =
          list
          |> Nx.tensor(type: type)
          |> Nx.to_binary()

        binary_backend_binary =
          list
          |> Nx.tensor(type: type, backend: Nx.BinaryBackend)
          |> Nx.to_binary()

        assert exaf_binary == binary_backend_binary
      end
    end

    test "constant creation" do
      for type <- @supported_types do
        scalar = 1

        exaf_binary =
          scalar
          |> Nx.tensor(type: type)
          |> Nx.to_binary()

        binary_backend_binary =
          scalar
          |> Nx.tensor(type: type, backend: Nx.BinaryBackend)
          |> Nx.to_binary()

        assert exaf_binary == binary_backend_binary
      end
    end
  end
end
