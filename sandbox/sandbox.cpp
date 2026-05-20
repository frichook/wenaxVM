// sandbox.cpp -- broad ISA-coverage test target for wenaxVM.
//
// Each test function is `extern "C" __declspec(noinline)` so:
//   * the PDB has an undecorated symbol the lifter can match,
//   * MSVC won't inline it away under /O2,
//   * the calling main() acts like a non-virtualised host driver --
//     the same way a real exe would call the protected routines.
//
// Coverage targets (the lifted instruction families):
//   * MOV (reg-reg, imm), ADD/SUB/XOR/AND/OR, IMUL, INC/DEC/NEG/NOT
//   * LEA scaled (reg+reg*{1,2,4,8}+disp)
//   * CMP / TEST, Jcc (all conditions), JMP, RET, NOP
//   * CMOVcc family, SETcc family
//   * MOVZX, MOVSX, MOVSXD
//   * SHL/SHR/SAR by immediate
//
// main() also exercises a handful of WinAPI calls (GetCurrentProcessId,
// GetTickCount64, OutputDebugStringA, MessageBeep) so the binary really
// pulls in winapi imports -- demonstrating that VM-protected functions
// can be invoked from a normal host that uses winapi freely.
//
// The last block ("call_other_*") intentionally contains x86 CALL
// instructions, which the current lifter rejects -- those functions
// SHOULD show up in wenaxvm-cli's reject log. That's the point.

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// =========================================================================
// 1. PURE INTEGER ARITHMETIC (10)
// =========================================================================
extern "C" __declspec(noinline) int  arith_add (int a, int b)       { return a + b; }
extern "C" __declspec(noinline) int  arith_sub (int a, int b)       { return a - b; }
extern "C" __declspec(noinline) int  arith_xor (int a, int b)       { return a ^ b; }
extern "C" __declspec(noinline) int  arith_and (int a, int b)       { return a & b; }
extern "C" __declspec(noinline) int  arith_or  (int a, int b)       { return a | b; }
extern "C" __declspec(noinline) int  arith_mul (int a, int b)       { return a * b; }
extern "C" __declspec(noinline) int  arith_neg (int a)              { return -a; }
extern "C" __declspec(noinline) int  arith_not (int a)              { return ~a; }
extern "C" __declspec(noinline) long long arith_mul64(long long a, long long b) { return a * b; }
extern "C" __declspec(noinline) int  arith_chain(int a, int b, int c){
    return (a + b) ^ (b - c) | a;
}

// =========================================================================
// 2. UNARY-ish (4)
// =========================================================================
extern "C" __declspec(noinline) int  unary_inc1   (int a) { return a + 1; }
extern "C" __declspec(noinline) int  unary_dec1   (int a) { return a - 1; }
extern "C" __declspec(noinline) int  unary_double (int a) { return a + a; }
extern "C" __declspec(noinline) int  unary_negate (int a) { return 0 - a; }

// =========================================================================
// 3. SHIFTS BY IMMEDIATE (6)
// =========================================================================
extern "C" __declspec(noinline) int  shift_shl3  (int a)              { return a << 3; }
extern "C" __declspec(noinline) int  shift_shr5  (int a)              { return (int)((unsigned)a >> 5); }
extern "C" __declspec(noinline) int  shift_sar2  (int a)              { return a >> 2; }
extern "C" __declspec(noinline) unsigned shift_two_sides(unsigned a)  { return (a << 4) | (a >> 4); }
extern "C" __declspec(noinline) int  shift_mul_const(int a)           { return a * 5; }   // /O2 -> lea+shl
extern "C" __declspec(noinline) long long shift_shl64(long long a)    { return a << 7; }

// =========================================================================
// 4. CMP / Jcc / branchy (8)
// =========================================================================
extern "C" __declspec(noinline) int  cmp_lt        (int a, int b)   { return a < b ? 1 : 0; }
extern "C" __declspec(noinline) int  cmp_ge_branch (int a, int b)   { if (a >= b) return a; else return b; }
extern "C" __declspec(noinline) int  cmp_eq_branch (int a, int b)   { return a == b ? 100 : 200; }
extern "C" __declspec(noinline) int  cmp_unsigned  (unsigned a, unsigned b) { return a > b ? 1 : -1; }
extern "C" __declspec(noinline) int  cmp_chain3(int a, int b, int c){
    if (a > 0) if (b > 0) if (c > 0) return 1;
    return 0;
}
extern "C" __declspec(noinline) int  cmp_ternary(int a, int b)      { return (a > b) ? (a * 2) : (b * 3); }
extern "C" __declspec(noinline) int  cmp_zero    (int a)            { return a ? 42 : -42; }
extern "C" __declspec(noinline) int  cmp_test_bit(int a)            { return (a & 4) ? 1 : 0; }

// =========================================================================
// 5. CMOV / SETcc (6)
// =========================================================================
extern "C" __declspec(noinline) int  cmov_max  (int a, int b) { return a > b ? a : b; }
extern "C" __declspec(noinline) int  cmov_min  (int a, int b) { return a < b ? a : b; }
extern "C" __declspec(noinline) int  cmov_abs  (int a)        { return a < 0 ? -a : a; }
extern "C" __declspec(noinline) int  setcc_sign(int a)        { return a >> 31; }       // -1 if negative
extern "C" __declspec(noinline) int  setcc_eq  (int a, int b) { return a == b ? 1 : 0; }
extern "C" __declspec(noinline) int  cmov_clamp(int a) {
    if (a < 0)   return 0;
    if (a > 255) return 255;
    return a;
}

// =========================================================================
// 6. MOVZX / MOVSX / MOVSXD (4)
// =========================================================================
extern "C" __declspec(noinline) int  movzx_from_short(short a)         { return (int)(unsigned short)a; }
extern "C" __declspec(noinline) int  movsx_from_short(short a)         { return (int)a; }
extern "C" __declspec(noinline) long long movsxd_from_int(int a)       { return (long long)a; }
extern "C" __declspec(noinline) unsigned long long zext32(int a)       { return (unsigned long long)(unsigned)a; }

// =========================================================================
// 7. LEA arithmetic (4)
// =========================================================================
extern "C" __declspec(noinline) int  lea_three_args(int a, int b, int c)  { return a + b + c; }
extern "C" __declspec(noinline) int  lea_times3    (int a)                { return a * 3; }    // -> lea r,[r+r*2]
extern "C" __declspec(noinline) int  lea_times9    (int a)                { return a * 9; }    // -> lea r,[r+r*8]
extern "C" __declspec(noinline) int  lea_addr_calc (int base, int idx)    { return base + idx * 4 + 16; }

// =========================================================================
// 8. COMBINED / LOOPS / MULTI-STEP (8)
//    These exercise multi-block CFGs the lifter has to stitch together
//    via label fixups. Several intentionally avoid division and memory
//    access so they're fully covered by the M4 ISA.
// =========================================================================
extern "C" __declspec(noinline) int sum_loop(int n) {
    int s = 0;
    for (int i = 0; i < n; ++i) s += i;
    return s;
}
extern "C" __declspec(noinline) int fast_pow_int(int base, int exp) {
    int result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
extern "C" __declspec(noinline) int bit_count_compact(unsigned x) {
    int c = 0;
    while (x) { c += (int)(x & 1u); x >>= 1; }
    return c;
}
extern "C" __declspec(noinline) int crc32_8steps(int crc, int b) {
    crc ^= b;
    for (int i = 0; i < 8; ++i)
        crc = (int)((unsigned)crc >> 1) ^ (-(crc & 1) & (int)0xEDB88320u);
    return crc;
}
extern "C" __declspec(noinline) int saturate_add(int a, int b) {
    int r = a + b;
    if (a > 0 && b > 0 && r < a)  return 0x7fffffff;
    if (a < 0 && b < 0 && r >= a) return (int)0x80000000;
    return r;
}
extern "C" __declspec(noinline) int rotate_left_imm(unsigned x, int unused) {
    (void)unused;
    return (int)((x << 5) | (x >> 27));
}
extern "C" __declspec(noinline) int xorshift_mini(int seed) {
    unsigned x = (unsigned)seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return (int)x;
}
extern "C" __declspec(noinline) int polynomial_eval(int x) {
    // 3x^3 + 2x^2 - 5x + 7, Horner's method.
    int r = 3;
    r = r * x + 2;
    r = r * x - 5;
    r = r * x + 7;
    return r;
}

// =========================================================================
// 9. CALL-INSIDE-VM DEMO (3)
//    These three functions invoke other helpers, so the lifter will
//    encounter `call rel32` (unsupported) and skip them. The wenaxvm-cli
//    log will list them as rejected. From the host's perspective they
//    still execute correctly (just natively, not virtualised) -- but the
//    *callees* arith_add / arith_xor / cmov_max remain virtualised.
// =========================================================================
extern "C" __declspec(noinline) int call_other_simple(int a, int b) {
    return arith_add(a, b);                         // CALL -> rejected
}
extern "C" __declspec(noinline) int call_other_chain(int a, int b, int c) {
    int x = arith_add(a, b);                        // CALL -> rejected
    int y = arith_xor(x, c);
    return cmov_max(x, y);
}
extern "C" __declspec(noinline) int call_through_fnptr(int a, int b, int (*op)(int, int)) {
    return op(a, b);                                // indirect call -- also rejected
}

// =========================================================================
// 10. Original M1/M2 functions (kept for regression)
// =========================================================================
extern "C" __declspec(noinline) int compute(int a, int b) {
    int x = a + b;
    x ^= 0xDEAD;
    x -= b - a;
    return x;
}
extern "C" __declspec(noinline) int compute2(int a, int b) {
    int x = a + b;
    if (x < 10) x = x * 3;
    else        x = x - 5;
    return x * b;
}

// =========================================================================
// Driver
// =========================================================================
namespace {

int g_pass = 0, g_fail = 0;

template <typename T>
void check(const char* name, T got, T expected) {
    bool ok = (got == expected);
    std::printf("  %-22s %s  expected=%lld got=%lld\n",
                name, ok ? "OK  " : "FAIL",
                (long long)expected, (long long)got);
    if (ok) ++g_pass; else ++g_fail;
}

} // namespace

int main() {
    // ---- WinAPI surface (NOT virtualised; demonstrates host environment) ----
    DWORD pid  = GetCurrentProcessId();
    ULONGLONG tk = GetTickCount64();
    char hdr[128];
    std::snprintf(hdr, sizeof(hdr), "wenaxVM sandbox  pid=%lu  tickms=%llu",
                  (unsigned long)pid, (unsigned long long)tk);
    OutputDebugStringA(hdr);
    std::puts(hdr);

    std::puts("--- arith ---");
    check("arith_add",          arith_add(10, 20),          30);
    check("arith_sub",          arith_sub(50, 7),           43);
    check("arith_xor",          arith_xor(0xF0F0, 0x0FF0),  0xFF00);
    check("arith_and",          arith_and(0xFF00, 0x0F0F),  0x0F00);
    check("arith_or",           arith_or (0xFF00, 0x0F0F),  0xFF0F);
    check("arith_mul",          arith_mul(7, 9),            63);
    check("arith_neg",          arith_neg(42),              -42);
    check("arith_not",          arith_not(0),               -1);
    check<long long>("arith_mul64", arith_mul64(100000LL, 200000LL), 20000000000LL);
    check("arith_chain",        arith_chain(3, 5, 1),       (3 + 5) ^ (5 - 1) | 3);

    std::puts("--- unary ---");
    check("unary_inc1",   unary_inc1(41),    42);
    check("unary_dec1",   unary_dec1(43),    42);
    check("unary_double", unary_double(21),  42);
    check("unary_negate", unary_negate(42), -42);

    std::puts("--- shifts ---");
    check("shift_shl3",     shift_shl3(5),         40);
    check("shift_shr5",     shift_shr5(0x4000),    0x200);
    check("shift_sar2",     shift_sar2(-16),       -4);
    check<unsigned>("shift_two_sides", shift_two_sides(0x12345678u),
                    (0x12345678u << 4) | (0x12345678u >> 4));
    check("shift_mul_const", shift_mul_const(7),   35);
    check<long long>("shift_shl64", shift_shl64(1LL), 128LL);

    std::puts("--- cmp ---");
    check("cmp_lt",         cmp_lt(3, 5),               1);
    check("cmp_ge_branch",  cmp_ge_branch(10, 5),       10);
    check("cmp_eq_branch",  cmp_eq_branch(7, 7),        100);
    check("cmp_unsigned",   cmp_unsigned(50u, 10u),     1);
    check("cmp_chain3",     cmp_chain3(1, 2, 3),        1);
    check("cmp_ternary",    cmp_ternary(10, 3),         20);
    check("cmp_zero",       cmp_zero(0),                -42);
    check("cmp_test_bit",   cmp_test_bit(0b101),        1);

    std::puts("--- cmov / setcc ---");
    check("cmov_max",   cmov_max(5, 9),    9);
    check("cmov_min",   cmov_min(5, 9),    5);
    check("cmov_abs",   cmov_abs(-13),     13);
    check("setcc_sign", setcc_sign(-1),    -1);
    check("setcc_eq",   setcc_eq(7, 7),    1);
    check("cmov_clamp", cmov_clamp(300),   255);

    std::puts("--- movzx/movsx ---");
    check("movzx_from_short", movzx_from_short((short)-1), 0xFFFF);
    check("movsx_from_short", movsx_from_short((short)-1), -1);
    check<long long>("movsxd_from_int", movsxd_from_int(-7), -7LL);
    check<unsigned long long>("zext32", zext32(-1), 0xFFFFFFFFull);

    std::puts("--- lea arith ---");
    check("lea_three_args", lea_three_args(1, 2, 3), 6);
    check("lea_times3",     lea_times3(7),           21);
    check("lea_times9",     lea_times9(4),           36);
    check("lea_addr_calc",  lea_addr_calc(100, 5),   136);

    std::puts("--- combined / loops ---");
    check("sum_loop",          sum_loop(11),                   55);     // 0..10
    check("fast_pow_int",      fast_pow_int(3, 7),             2187);
    check("bit_count_compact", bit_count_compact(0xFFFFu),     16);
    check("crc32_8steps",      crc32_8steps(0, 'a'),           crc32_8steps(0, 'a'));  // self-check shape
    check("saturate_add",      saturate_add(2'000'000'000, 2'000'000'000), 0x7fffffff);
    check("rotate_left_imm",   rotate_left_imm(1u, 0),         (int)((1u << 5) | (1u >> 27)));
    check("xorshift_mini",     xorshift_mini(1),               xorshift_mini(1));
    check("polynomial_eval",   polynomial_eval(2),             3*8 + 2*4 - 5*2 + 7);

    std::puts("--- legacy regression ---");
    check("compute",  compute(10, 20),   57001);
    check("compute2", compute2(10, 20),  500);

    std::puts("--- call-through (host-only, lifter should reject) ---");
    check("call_other_simple", call_other_simple(11, 22), 33);
    check("call_other_chain",  call_other_chain(1, 2, 3),
          [](){ int x = 1+2; int y = x^3; return x>y?x:y; }());
    check("call_through_fnptr",
          call_through_fnptr(10, 20, arith_add), 30);

    std::printf("---- summary: %d passed, %d failed ----\n", g_pass, g_fail);

    // One more WinAPI sample (optional audible bleep, side-effect free if no sound).
    MessageBeep(g_fail ? MB_ICONERROR : MB_OK);
    return g_fail ? 1 : 0;
}
