local FMeasMatch, parent = torch.class('kttorch.FMeasMatch', 'nn.Module')

-- @param reference a vector of indiced corresponding to the sample reference (or human reference) 
function FMeasMatch:__init(references)
  parent.__init(self)
  self.references = references
end

function numOverlaps(txt1, txt2)
  tbl = {}
  if txt1:nElement() == 0 or txt2:nElement() ==0 then
    return 0
  end
  -- Adding txt1's tokens to a table
  for i =1,txt1:size(1) do
    if tbl[txt1[i]] == nil then
      tbl[i] = txt1[i]
    end
  end
  -- Iterate over txt2 and select the ones in tbl
  local overlaps = 0
  for i =1,txt2:size(1) do
    if tbl[txt2[i]] ~= nil then
      overlaps = overlaps +1
    end
  end
  return overlaps 
end

-- @param input One student essay at a time 
function FMeasMatch:updateOutput(input)
  input = input[input:ne(-1)]
  if input:nElement() == 0 then
    return 0 
  end
  -- Comparison with each reference
  local maxFMeasure = 0
  for j=1,self.references:size(1) do
    reference = self.references[j]
    reference = reference[reference:ne(-1)]
    local overlaps = numOverlaps(reference, input)
    -- Compute precision
    local precision = overlaps/input:nElement()
    -- Compute recall
    local recall = overlaps/reference:nElement()
    -- Compute f-measure
    local fmeasure = 0
    if precision ~= 0 or recall ~= 0 then
      fmeasure = (2 * precision * recall)/(precision + recall)
    else
      fmeasure = 0
    end
    if fmeasure > maxFMeasure then
      maxFMeasure = fmeasure
    end
  end
  return maxFMeasure 
end
