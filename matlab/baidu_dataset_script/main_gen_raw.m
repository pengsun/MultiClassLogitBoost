%% prepaer data
clear
% k: key value; d: date; c: count
[k,d,c] = textread('data.2012-2013.filtered.100','%s %d %f','delimiter','\t');
[kname, x, kid] = unique(k);
[dname, x, did] = unique(d);
clear k;
clear d;

ff = [];
for i = 1:length(kname)
    if strcmp(kname{i},'ss上证指数ss')
        ff(1) = i;
    end
    if strcmp(kname{i},'cc金融理财cc')
        ff(2) = i;
    end
end

m = full(sparse(did,kid,c));
f = find(m(:,ff(1)) > 0);
dat = m(f,:);
dnameall = dname;
dname = dnameall(f);
szzs = dat(:,ff(1)); % Y vector
jrlc = dat(:,ff(2));
[a,b] = size(dat);
ff1 = setdiff(1:b,ff);
dat = [dat(:,ff1)]; % X matrix

kname = kname(ff1);
%% X, Y
% data
X = dat;
X = X(1:end-1,:);
X = X';
X = single(X);
% target
Y = diff(szzs) > 0;
Y = Y(:)';
Y = single(Y);
%% train, test
nclass = 2;
[nvar,n] = size(X);
cat_mask = uint8(zeros(nvar,1));
%% save
save('S_raw.mat',...
  'X','Y',...
  'cat_mask', 'nclass', 'nvar');