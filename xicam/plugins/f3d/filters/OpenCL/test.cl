kernel void test(global const uchar* inputBuffer, global uchar* outputBuffer)
{

    int i = get_global_id(0);
    outputBuffer[i] = inputBuffer[i];
}