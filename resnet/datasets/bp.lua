--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  bp dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'
local cv = require 'cv'

local M = {}
local bpDataset = torch.class('resnet.bpDataset', M)

function bpDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function bpDataset:get(i)
   local path = ffi.string(self.imageInfo.imagePath[i]:data())

   local image = self:_loadImage(paths.concat(self.dir, path))
   local class = self.imageInfo.imageClass[i]   

   image = self:_preprocess(image)

   return {
      input = image,
      target = class,
   }
end

function bpDataset:_preprocess( img )
   --resize?
   new_img = cv.EqualizeHist(img) --histogram equalization
   new_img = new_img:add(-meanstd.mean)
   new_img = new_img:div(meanstd.std)   
   return new_img
end

function bpDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))
      input = image.decompress(b, 3, 'float')
   end

   return input
end

function bpDataset:size()
   return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of bp training images
-- histogram equalized
local meanstd = {
   mean = 127.5648, 
   std = 74.8659,
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}
-- --raw images
-- local meanstd = {
--    mean = {  }, 
--    std = { },
-- }
-- local pca = {
--    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
--    eigvec = torch.Tensor{
--       { -0.5675,  0.7192,  0.4009 },
--       { -0.5808, -0.0045, -0.8140 },
--       { -0.5836, -0.6948,  0.4203 },
--    },
-- }

function bpDataset:augment()
   if self.split == 'train' then
      -- local randNo = math.random(5)
      -- if randNo == 1 then
      --    return t.Compose{
      --    t.Translation(10)}
      -- elseif randNo == 2 then 
      --    return t.Compose{
      --    t.HorizontalFlip(1)}
      -- elseif randNo == 3 then
      --    return t.Compose{
      --    t.Rotation(1)}
      -- elseif randNo == 4 then
      --    return t.Compose{
      --    t.Rotation(-1)}
      -- else 
            return t.Compose{
            t.Zoom(0.5),
            t.HorizontalFlip(0.5),
            t.VerticalFlip(0.5),
            t.Translation(0.5),
            t.Rotation(0.5)
         -- t.RandomSizedCrop(224),
         -- t.ColorJitter({brightness = 0.4,contrast = 0.4,saturation = 0.4,}),
         -- t.Lighting(0.1, pca.eigval, pca.eigvec),
            }
      elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         -- t.Scale(256),
         -- t.ColorNormalize(meanstd),
         -- Crop(224),
         t.NoChange()
      }
   else
      error('invalid split: ' .. self.split)
   end
end



return M.bpDataset
