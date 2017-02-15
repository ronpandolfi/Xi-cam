kernel void sum(
    global const float *a_g, global const float *b_g, global float *res_g)
{
  int gid = get_global_id(1)
  res_g[gid] = gid;
}