%%function for PSR calculation
function [out] = PsrCalculation(Corr)
peak = max(max(Corr));
[h1,h2] = find(Corr == peak);

a = size(Corr,1);
b = size(Corr,2);
        sumtot = sum(sum(Corr));
      %  sumpeak = sum(sum(Corr((h1):(h1 + 10),(h2):(h2 + 10))));
     %   sumlobe = sumtot - sumpeak;
     %   n = a*b - 11*11;
      %  mn = sumlobe/n;
        
       Corr((h1):(h1 + 10),(h2):(h2 + 10)) = 0;
    %  Mask = Corr((h1-5):(h1+5), (h2-5):(h2+5));
      cnt = 1;
        for ic = 1:size(Corr,1)
            for ik = 1:size(Corr,2)
                if Corr(ic,ik) == 0
                
                else 
                    Annular(cnt,:) = Corr(ic,ik);
                    cnt = cnt + 1;
                end
            end   
        end
        mn = mean(Annular);
        st = std(Annular);
        psr = (peak - mn)/st;

out = psr;
end