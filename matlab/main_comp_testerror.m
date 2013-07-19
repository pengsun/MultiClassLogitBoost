%%
% name = 'poker100k';
name = 'pendigits';
% name = 'optdigits';
% name = 'M-Basic';
% name = 'isolet';

algoname1 = 'VTLogitBoost';
dir_root1 = fullfile('.\rst',algoname1);
fn1 = 'T300_v1.0e-001_J20_ns1.mat';

algoname2 = 'VTTCLogitBoost';
dir_root2 = fullfile('.\rst',algoname2);
fn2 = 'T300_lambda0_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-007_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-006_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-005_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-004_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-003_J20_ns1.mat';
% fn2 = 'T300_lambda1.0e-002_J20_ns1.mat';

%% load
ffn1 = fullfile(dir_root1,name,fn1);
tmp = load(ffn1);
it1 = tmp.it;
err_it1 = tmp.err_it;
abs_grad1 = tmp.abs_grad;
F1 = tmp.F;
num_it1 = tmp.num_it;
clear tmp;

ffn2 = fullfile(dir_root2,name,fn2);
tmp = load(ffn2);
it2 = tmp.it;
err_it2 = tmp.err_it;
abs_grad2 = tmp.abs_grad;
F2 = tmp.F;
num_it2 = tmp.num_it;

%% error
figure('name',name); title error; hold on;
% plot(it1,err_it1, 'color','r','marker','.');
% plot(it2,err_it2, 'color','b','marker','.');
plot(it1,err_it1, 'color','r','lineWidth', 2, 'marker','s');
plot(it2,err_it2, 'color','b','lineWidth', 2, 'marker','.');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2]);
set(h,'Interpreter','none'); 
grid on; hold off; 

% tune the appearence
ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/2);
set(gca, 'ylim', [0,500]);

%% grad
figure('name',name);  title grad; hold on;
% plot(it1, log10( abs_grad1(it1) ), 'color','r','marker','.');
% plot(it2, log10( abs_grad2(it2) ), 'color','b','marker','.');
plot(it1, abs_grad1(it1), 'color','r','marker','s');
plot(it2, abs_grad2(it2), 'color','b','marker','.');
h = legend(...
    [algoname1,'-',fn1],...
    [algoname2,'-',fn2]);
set(h, 'Interpreter','none'); 
grid on;  hold off;

% tune the appearence
% ylim = get(gca,'ylim');
% set(gca,'ylim',ylim/3);
% 
% %% margin
% dataset = fullfile('.\dataset', [name, '.mat']);
% tmp = load(dataset);
% Y = tmp.Ytr + 1;
% [~, n] = size(Y);
% 
% % margin1
% margin1 = zeros(1, n);
% for i=1:n
%     margin1(i) = F1( Y(i), i);
%     F1( Y(i), i) = - inf;
%     margin1(i) =  margin1(i) - max(F1(:, i));
% end
% 
% % margin2
% margin2 = zeros(1, n);
% for i=1:n
%     margin2(i) = F2( Y(i), i);
%     F2( Y(i), i) = - inf;
%     margin2(i) =  margin2(i) - max(F2(:, i));
% end
% 
% % normalize
% maxM = max( max(margin1), max(margin2));
% minM = min( min(margin1), min(margin2));
% minM = abs(minM);
% maxM = max(maxM, minM);
% margin1 = margin1 ./ maxM ;
% margin2 = margin2 ./ maxM;
% 
% % plot cdf
% % [pdf1, xi1] = ksdensity(margin1, 'function','cdf');
% % [pdf2, xi2] = ksdensity(margin2, 'function','cdf');
% xi = linspace(-1, 1, 200);
% cdf1 = ksdensity(margin1, xi, 'function','cdf');
% cdf2 = ksdensity(margin2, xi, 'function','cdf');
% 
% figure('name',name); hold on;
% xlabel('margin'); ylabel('cumulate frequency');
% 
% plot(xi, cdf1, 'color','r');
% plot(xi, cdf2, 'color','b');
% 
% legend('AOSO','AOSO1' ); grid on; hold off;
% set(gca,'ylim',[0, 1.2]);
% set(gca,'xlim',[0, 1.2]);
% % % plot pdf
% % [pdf1, xi1] = ksdensity(margin1);
% % [pdf2, xi2] = ksdensity(margin2);
% % 
% % figure('name',name); title margin(pdf); hold on;
% % plot(xi1,pdf1, 'color','r');
% % plot(xi2,pdf2, 'color','b');
% % legend('CBT','AOSO'); grid on; hold off;
% 
% %% best result
% fprintf('-------------\n');
% fprintf('dataset: %s\n\n', name);
% fprintf('best result:\n');
% [err1best,it1best] = min(err_it1);
% [err2best,it2best] = min(err_it2);
% fprintf('AOSO: %d @ %d\n', err1best, it1(it1best) );
% fprintf('AOSO1: %d @ %d\n\n', err2best, it2(it2best));
% 
% %% last result
% fprintf('last result:\n');
% fprintf('AOSO: %d @ %d\n', err_it1(end), it1(end));
% fprintf('AOSO1: %d @ %d\n\n', err_it2(end), it2(end));
% 
% % itNum = uint32(num_it2/num_it1*2000);
% % fprintf('AOSO: %d @ %d\n', err_it1(end), it1(itNum));
% % fprintf('AOSO1: %d @ %d\n\n', err_it2(end), it2(end));
% 
% %% tree depth
% fprintf('tree depth:\n');
% ffn1 = fullfile(dir_root1,name,'treeDepth.mat');
% load(ffn1);
% fprintf('AOSO average max-depth: %.4f \n', avr_maxDepth);
% fprintf('AOSO average avr-depth: %.4f \n', avr_avrDepth);
% 
% ffn2 = fullfile(dir_root2,name,'treeDepth.mat');
% load(ffn2);
% fprintf('AOSO1 average max-depth: %.4f \n', avr_maxDepth);
% fprintf('AOSO1 average avr-depth: %.4f \n\n', avr_avrDepth);
% 
% clear;