--[[This script computes cosine similarity between
two vectors. --]]

require 'torch'   -- torch
require 'math' -- for square root functionality

function cosineSimilarity(vec1, vec2)
  if vec1:nElement() ~= vec2:nElement() then
    print "Vector sizes are different!"
    return
  end
  local sum = 0
  local sumvec1 = 0
  local sumvec2 = 0
  for i =1,vec1:nElement() do
    sum = sum + (vec1[i] * vec2[i])
    sumvec1 = sumvec1 + (vec1[i] * vec1[i])
    sumvec2 = sumvec2 + (vec2[i] * vec2[i])
  end
  local cosine = sum/(math.sqrt(sumvec1) * math.sqrt(sumvec2))
  return cosine
end
