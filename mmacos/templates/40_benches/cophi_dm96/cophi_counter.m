classdef cophi_counter < handle
%COPHI_COUNTER  A measurement counter shared by the COPHI loop instrument's
%   handles (the LO drift advances per measurement, which an anonymous function
%   cannot remember).  n = measurements taken so far.
    properties
        n = 0
    end
end
