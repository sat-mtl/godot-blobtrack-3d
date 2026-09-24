bool posValid(vec3 p)
{
  // retourne false si une composante est NaN, Infinity ou plus grand en valeur absolue que 1.0e18
  return !any(isnan(p)) && !any(isinf(p)) && all(lessThan(abs(p), vec3(1.0e18)));
}

ivec3 cellOf(vec3 p, float invCellSize)
{
  // Clamped so the float-to-int conversion stays defined for far-flung
  // coordinates. Points that far apart never share a cell anyway.
  return ivec3(floor(clamp(p * invCellSize, vec3(-1.0e9), vec3(1.0e9))));
}

uint bucketOf(ivec3 c, uint tableSize)
{
  // cette fonction semble venir de ce papier https://matthias-research.github.io/pages/publications/tetraederCollision.pdf
  // peut être voir https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
  // vu que le papier ne semble pas certain que c'est si optimal que ça.
  // j'ai lu sur stackoverflow aussi que les hash functions sans bitshifts sont souvent
  // de moins bonne qualité.
  return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u)
      & (tableSize - 1u);
}
