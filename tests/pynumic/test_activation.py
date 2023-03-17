# class Activation:
#     def test_activation(self):
#         assert False


def test_get_activation() -> None:
    assert False


def test_get_derivative() -> None:
    assert False

# func TestActivation(t *testing.T) {
# 	type args struct {
# 		value pkg.FloatType
# 		mode  uint8
# 	}
# 	tests := []struct {
# 		name string
# 		args
# 		want pkg.FloatType
# 	}{
# 		{
# 			name: "#1_LINEAR",
# 			args: args{.1, LINEAR},
# 			want: .1,
# 		},
# 		{
# 			name: "#2_RELU",
# 			args: args{.1, RELU},
# 			want: .1,
# 		},
# 		{
# 			name: "#3_RELU",
# 			args: args{-.1, RELU},
# 			want: 0,
# 		},
# 		{
# 			name: "#4_LEAKYRELU",
# 			args: args{.1, LEAKYRELU},
# 			want: .1,
# 		},
# 		{
# 			name: "#5_LEAKYRELU",
# 			args: args{-.1, LEAKYRELU},
# 			want: -.001,
# 		},
# 		{
# 			name: "#6_SIGMOID",
# 			args: args{.1, SIGMOID},
# 			want: .52497918747894,
# 		},
# 		{
# 			name: "#7_TANH",
# 			args: args{.1, TANH},
# 			want: .099668,
# 		},
# 		{
# 			name: "#8_default",
# 			args: args{.1, 255},
# 			want: .52497918747894,
# 		},
# 	}
#
# 	for _, tt := range tests {
# 		t.Run(tt.name, func(t *testing.T) {
# 			if got := Activation(tt.value, tt.mode); got != tt.want {
# 				t.Errorf("Activation() = %g, want %g", got, tt.want)
# 			}
# 		})
# 	}
# }
#
# func TestDerivative(t *testing.T) {
# 	type args struct {
# 		value pkg.FloatType
# 		mode  uint8
# 	}
# 	tests := []struct {
# 		name string
# 		args
# 		want pkg.FloatType
# 	}{
# 		{
# 			name: "#1_LINEAR",
# 			args: args{.1, LINEAR},
# 			want: 1,
# 		},
# 		{
# 			name: "#2_RELU",
# 			args: args{.1, RELU},
# 			want: 1,
# 		},
# 		{
# 			name: "#3_RELU",
# 			args: args{-.1, RELU},
# 			want: 0,
# 		},
# 		{
# 			name: "#4_LEAKYRELU",
# 			args: args{.1, LEAKYRELU},
# 			want: 1,
# 		},
# 		{
# 			name: "#5_LEAKYRELU",
# 			args: args{-.1, LEAKYRELU},
# 			want: .01,
# 		},
# 		{
# 			name: "#6_SIGMOID",
# 			args: args{.1, SIGMOID},
# 			want: .089999996,
# 		},
# 		{
# 			name: "#7_TANH",
# 			args: args{.1, TANH},
# 			want: .99,
# 		},
# 		{
# 			name: "#8_default",
# 			args: args{.1, 255},
# 			want: .089999996,
# 		},
# 	}
#
# 	for _, tt := range tests {
# 		t.Run(tt.name, func(t *testing.T) {
# 			if got := Derivative(tt.value, tt.mode); got != tt.want {
# 				t.Errorf("Derivative() = %g, want %g", got, tt.want)
# 			}
# 		})
# 	}
# }
