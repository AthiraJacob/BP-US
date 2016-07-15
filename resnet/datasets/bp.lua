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
require 'cv.imgcodecs'
require 'cv.imgproc'


local M = {}
local bpDataset = torch.class('resnet.bpDataset', M)

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
   	
   local newImg = torch.Tensor(1,image:size(1),image:size(2))
   newImg[{{1}}] = image
   -- print(newImg:size())

   return {
      input = newImg,
      target = class,
   }
end

function bpDataset:_preprocess( img )
   --resize?
   cv.equalizeHist(img,img) --histogram equalization
   -- print(img:size())
   new_img = img:add(-meanstd.mean)
   new_img = new_img:div(meanstd.std)  
   new_img = self:resize(new_img,224)
   return new_img
end

function bpDataset:resize(img,size)
	interpolation = interpolation or 'bicubic'
	local w, h = img:size(2), img:size(1)
	size = size+1
      if w < h then
         newImg = image.scale(img, size, h/w * size, interpolation)
      else
         newImg = image.scale(img, w/h * size, size, interpolation)
     end
     size = size-1
     w = math.ceil((newImg:size(2) - size)/2)
     h = math.ceil((newImg:size(1) - size)/2)
     return image.crop(newImg, w, h, w + size, h + size)
 end



function bpDataset:_loadImage(path)
   local ok, input = pcall(function()
      return cv.imread{path,cv.IMREAD_GRAYSCALE}
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
