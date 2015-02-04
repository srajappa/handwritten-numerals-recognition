function writetofile(y)
    fid = fopen('classes_lr.txt','wt');

    for ii = 1:size(y,1)
        fprintf(fid,'%g\t',y(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
end