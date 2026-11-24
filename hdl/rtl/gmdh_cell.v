// Reusable 2-input affine GMDH Processing Element (Verilog-2001).
// Computes: y_next = clamp(round_shift((W0 << FRAC_W) + W1*x1 + W2*x2, FRAC_W))
// ACT_TYPE: 0 = linear (saturate to DATA_W), 1 = Hardtanh [ACT_MIN, ACT_MAX], 2 = ReLU [0, Q_MAX].

`timescale 1ns/1ps

module gmdh_cell #(
    parameter integer DATA_W        = 12,
    parameter integer FRAC_W        = 7,
    parameter integer ACC_W         = 28,
    parameter signed [DATA_W-1:0] W0 = 0,
    parameter signed [DATA_W-1:0] W1 = 0,
    parameter signed [DATA_W-1:0] W2 = 0,
    parameter integer ACT_TYPE      = 1,
    parameter signed [DATA_W-1:0] ACT_MIN = -256,
    parameter signed [DATA_W-1:0] ACT_MAX =  256,
    parameter integer ROUND_NEAREST = 1
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire signed [DATA_W-1:0] x1,
    input  wire signed [DATA_W-1:0] x2,
    output reg  signed [DATA_W-1:0] y
);
    localparam signed [ACC_W-1:0] Q_MIN = -(1 <<< (DATA_W - 1));
    localparam signed [ACC_W-1:0] Q_MAX =  (1 <<< (DATA_W - 1)) - 1;
    localparam signed [ACC_W-1:0] ACT_MIN_EXT = ACT_MIN;
    localparam signed [ACC_W-1:0] ACT_MAX_EXT = ACT_MAX;
    localparam signed [ACC_W-1:0] CLAMP_LO = (ACT_TYPE == 1) ? ACT_MIN_EXT : ((ACT_TYPE == 2) ? 0 : Q_MIN);
    localparam signed [ACC_W-1:0] CLAMP_HI = (ACT_TYPE == 1) ? ACT_MAX_EXT : Q_MAX;
    localparam signed [ACC_W-1:0] ROUND_BIAS = (ROUND_NEAREST != 0 && FRAC_W > 0) ? (1 <<< (FRAC_W - 1)) : 0;

    wire signed [ACC_W-1:0] b_ext  = W0;
    wire signed [ACC_W-1:0] w1_ext = W1;
    wire signed [ACC_W-1:0] w2_ext = W2;
    wire signed [ACC_W-1:0] x1_ext = x1;
    wire signed [ACC_W-1:0] x2_ext = x2;

    wire signed [ACC_W-1:0] acc     = (b_ext <<< FRAC_W) + (w1_ext * x1_ext) + (w2_ext * x2_ext);
    wire signed [ACC_W-1:0] shifted = (acc + ROUND_BIAS) >>> FRAC_W;
    wire signed [ACC_W-1:0] clamped =
        (shifted < CLAMP_LO) ? CLAMP_LO :
        (shifted > CLAMP_HI) ? CLAMP_HI : shifted;

    always @(posedge clk) begin
        if (!rst_n)
            y <= {DATA_W{1'b0}};
        else
            y <= clamped[DATA_W-1:0];
    end
endmodule
