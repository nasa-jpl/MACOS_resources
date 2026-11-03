% MK_LADDER_FIG  The descent's one page: what mirrors buy, against what the
% requirement set asks for.  Log scale on purpose -- the story is two orders
% of magnitude, and a linear axis would draw every rung flat on the floor.
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));  up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing')); addpath(fullfile(up,'wall'));

% full-solve rungs (the ascent) + the committed parent
Nfull   = [4 5 6 7];
wfeFull = [10407.0 10774.9 9137.4 7894.1];
blur    = [157.0 332.3 720.5 704.6];
% wavefront-only floors: the pupil requirement abandoned entirely
Nfl   = [4 5 6 7];
wfeFl = [3841.8 8077.4 5689.0 3424.2];

f = figure('Position',[60 60 1180 480],'Color','w','Visible','off');
tl = tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');

ax = nexttile(tl); hold(ax,'on');
plot(ax,Nfull,wfeFull,'-o','LineWidth',2,'MarkerSize',7,'MarkerFaceColor','auto');
plot(ax,Nfl,wfeFl,'--s','LineWidth',2,'MarkerSize',7);
yline(ax,71,'r-','LineWidth',2);
set(ax,'YScale','log');  grid(ax,'on');  box(ax,'on');
xlabel(ax,'powered mirrors, N');  ylabel(ax,'WFE rung 2, max over the field box  (nm)');
xlim(ax,[3.6 7.4]);  ylim(ax,[40 2e4]);
legend(ax,{'full requirement set','wavefront ONLY (pupil abandoned)', ...
           'the 71 nm target'},'Location','southwest','Box','off');
title(ax,'three extra mirrors buy 11 %; the target needs 48x');

ax2 = nexttile(tl); hold(ax2,'on');
yyaxis(ax2,'left');
plot(ax2,Nfull,wfeFull/71,'-o','LineWidth',2,'MarkerSize',7,'MarkerFaceColor','auto');
ylabel(ax2,'wavefront, as a multiple of its target');
set(ax2,'YScale','log');
yyaxis(ax2,'right');
plot(ax2,Nfull,blur/47,'-^','LineWidth',2,'MarkerSize',7);
ylabel(ax2,'pupil blur, as a multiple of its target');
yline(ax2,1,'k:','LineWidth',1.5);
xlabel(ax2,'powered mirrors, N');  grid(ax2,'on');  box(ax2,'on');
xlim(ax2,[3.6 7.4]);
legend(ax2,{'wavefront / 71 nm','pupil blur / 47 um','target'}, ...
       'Location','northwest','Box','off');
title(ax2,'and the pupil pays for what the wavefront gains');

annotation(f,'textbox',[0.02 0.955 0.96 0.04],'String', ...
  ['the afocal4 descent: no mirror count in this family reaches the ' ...
   'requirement set'],'HorizontalAlignment','center','EdgeColor','none', ...
   'FontWeight','bold','FontSize',11);
tl.OuterPosition = [0 0 1 0.94];
png = fullfile(here,'afocal4_descent_ladder.png');
exportgraphics(f,png,'Resolution',150);  close(f);
fprintf('wrote %s\n', png);
exit(0);
