local TextOverlaps, parent = torch.class('kttorch.TextOverlaps', 'nn.Module')

--[[
--]]
-- @param reference a vector of indiced corresponding to the sample reference (or human reference)
-- @param proxyVecPos contains the position of the proxy vector to use when the module returns an 
-- empty tensor (if there are no overlaps between the texts or some other reason).
function TextOverlaps:__init(references, proxyVecPos)
  parent.__init(self)
  self.references = references
  self.proxyVecPos = proxyVecPos
end

-- @param input 
function TextOverlaps:updateOutput(input)
  if input:nElement() == 0 then
    return torch.Tensor(1):fill(self.proxyVecPos)
  end
  tbl = {}
  bestMatchinput = torch.Tensor()
  -- Comparison with each reference
  for j=1,self.references:size(1) do 
    reference = self.references[j]
    -- Adding the reference's tokens to a table
    reference = reference[reference:ne(-1)]
    if reference:nElement() ~= 0 then
      for i =1,reference:size(1) do
        if tbl[reference[i]] == nil then
          tbl[i] = reference[i]
        end
      end
      -- Iterate over the input and select the ones in tbl
      for i =1,input:size(1) do
        if tbl[input[i]] == nil then
          input[i] = -1 --Dropping tokens not in the table
        end
      end
      result = input[input:ne(-1)]
      --Selecting the vector that has a better match with one of the references
      if bestMatchinput:nElement() == 0 or (result:nElement() ~= 0 and result:size(1) > bestMatchinput:size(1)) then
        bestMatchinput = result
      end
    end
  end
  if bestMatchinput:nElement() == 0 then
    return torch.Tensor(1):fill(self.proxyVecPos)
  else
    return bestMatchinput
  end
end
