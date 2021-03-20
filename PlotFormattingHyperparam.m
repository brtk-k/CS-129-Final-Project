% Auxillary plot formatting function for the hyperparameter tuning
% visualization
function PlotFormattingHyperparam(plot_title, label_x, label_y, label_z, legendFlag, legLabels, legLoc, fontSize, ylabelFlag)
    grid on;
    set(gca,'TickLabelInterpreter','latex','FontSize',fontSize)
    title(plot_title,'Interpreter','latex','FontSize', fontSize);
    xlabel(label_x,'Interpreter','latex','FontSize',fontSize);
    if ylabelFlag
        ylabel(label_y,'Interpreter','latex','FontSize',fontSize);
    end
    zlabel(label_z,'Interpreter','latex','FontSize',fontSize);
    if legendFlag
        legend(legLabels,'Interpreter','latex','FontSize',fontSize - 4,...
               'location',legLoc);
    end
end