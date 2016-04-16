
%  Exploiting the Circulant Structure of Tracking-by-detection with Kernels
%
%  Main script for tracking, with a gaussian kernel.
%
%  Jo?o F. Henriques, 2012
%  http://www.isr.uc.pt/~henriques/


%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = './Suv';


%parameters according to the paper
padding = 1;					%extra area surrounding the target
output_sigma_factor = 1/16;		%spatial bandwidth (proportional to target)
sigma = 0.2;					%gaussian kernel bandwidth
lambda = 1e-2;					%regularization
interp_factor = 0.075;			%linear interpolation factor for adaptation
th = 0.35;                       %threshold for occlusion


%notation: variables ending with f are in the frequency domain.

%ask the user for the video
video_path = choose_video(base_path);
if isempty(video_path), return, end  %user cancelled
[img_files, pos, target_sz, resize_image, ground_truth, video_path] = ...
	load_video_info(video_path);


%window size, taking padding into account
sz = floor(target_sz * (1 + padding));

%desired output (gaussian shaped), bandwidth proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
y = exp(-0.5 / output_sigma^2 * (rs.^2 + cs.^2));
yf = fft2(y);

%store pre-computed cosine window
cos_window = hann(sz(1)) * hann(sz(2))';


time = 0;  %to calculate FPS
positions = zeros(numel(img_files), 2);  %to calculate precision
PsrVal = zeros(numel(img_files),1);
loopcount=0;
trackInit = false;
%trackedLocation = [];
for frame = 1:numel(img_files),
	%load image
    loopcount = 0;
	im = imread([video_path img_files{frame}]);
	if size(im,3) > 1,
		im = rgb2gray(im);
	end
	if resize_image,
		im = imresize(im, 0.5);
	end
	
	tic()
	
	%extract and pre-process subwindow
	x = get_subwindow(im, pos, sz, cos_window);

    if frame > 1,
		%calculate response of the classifier at all locations
		k = dense_gauss_kernel(sigma, x, z);
		response = real(ifft2(alphaf .* fft2(k)));   %(Eq. 9)
		PsrVal(frame) = PsrCalculation(response);
        %disp(PsrVal(frame))
		%target location is at the maximum response
		[row, col] = find(response == max(response(:)), 1);
        if (PsrVal(frame)< maxPsr*th)
        param = getDefaultParameters();
        initialLocation = computeInitialLocation(param, pos);
        %init Kalman filter
        kalmanFilter = configureKalmanFilter(param.motionModel,... 
        initialLocation, param.initialEstimateError,...
        param.motionNoise, param.measurementNoise);
        pos = predict(kalmanFilter);
        else
        pos = pos - floor(sz/2) + [row, col];
        end
	end
	
	%get subwindow at current estimated target position, to train classifer
	x = get_subwindow(im, pos, sz, cos_window);
	
	%Kernel Regularized Least-Squares, calculate alphas (in Fourier domain)
	k = dense_gauss_kernel(sigma, x);
	new_alphaf = yf ./ (fft2(k) + lambda);   %(Eq. 7)
	new_z = x;
	
	if frame == 1,  %first frame, train with a single image
		alphaf = new_alphaf;
		z = x;
         %init kalman
 
        param = getDefaultParameters();
        initialLocation = computeInitialLocation(param, pos);
        %init Kalman filter
        kalmanFilter = configureKalmanFilter(param.motionModel,... 
        initialLocation, param.initialEstimateError,...
        param.motionNoise, param.measurementNoise);
        predict(kalmanFilter);
        %trackedLocation = correct(kalmanFilter,pos);
        pos = correct(kalmanFilter,pos);
	else
		%subsequent frames, interpolate model
		alphaf = (1 - interp_factor) * alphaf + interp_factor * new_alphaf;
		z = (1 - interp_factor) * z + interp_factor * new_z;
    end
   
  %  trackInit = true;
    maxPsr=max(PsrVal);
	%save position and calculate FPS
	positions(frame,:) = pos;
   % if (PsrVal(frame)<maxPsr*th)
   % kalmanpos(frame,:) = trackedLocation;
   % else
   % kalmanpos(frame,:) = pos;
   % end
	time = time + toc();
    
	
	%visualization
	rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
 %   rect_position2 = [trackedLocation([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%     if ((frame)<maxPsr*th)
%         loopcount=loopcount+1;
%     end
%      disp(loopcount)
%      disp(PsrVal(frame))
    
	if frame == 1,  %first frame, create GUI
        h=figure();
        im_handle = imshow(im, 'Border','tight', 'InitialMag',200);
        rect_handle = rectangle('Visible', 'off','Position',rect_position, 'EdgeColor','b');
        rect_handle2 = rectangle('Visible', 'off','Position',rect_position, 'EdgeColor','r');
    end
    
    if frame>1,
        if (loopcount == 0)
            %%if normal
            set(rect_handle, 'Visible', 'on','Position', rect_position)
           set(rect_handle2, 'Visible', 'off','Position', rect_position)
        end
        if (PsrVal(frame)<maxPsr*th)  %%if occlusion
            %subsequent frames, update GUI
            fprintf('occluded')
           % trackedLocation = predict(kalmanFilter);
            loopcount=loopcount+1;
            set(rect_handle, 'Visible', 'off','Position', rect_position)
            set(rect_handle2, 'Visible', 'on','Position', rect_position)
        end
     
    set(im_handle, 'CData', im)
    
    drawnow
    %pause(0.05)  %uncomment to run slower
    if ~ishandle(h)
        return
    end
    end
end


if resize_image, positions = positions * 2; end

disp(['Frames-per-second: ' num2str(numel(img_files) / time)])

%show the precisions plot
show_precision(positions, ground_truth, video_path)
%show_precision(kalmanpos, ground_truth, video_path)