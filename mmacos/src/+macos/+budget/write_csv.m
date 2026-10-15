function write_csv(T, path)
%WRITE_CSV  Write a budget roll-up table to CSV.
%   macos.budget.write_csv(T, PATH) writes the table from
%   macos.budget.rollup to PATH via writetable.
%
%   See also macos.budget.rollup, macos.budget.write_xlsx.
writetable(T, path);
fprintf('wrote %s\n', path);
end
